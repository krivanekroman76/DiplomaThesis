import warnings
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import json
from tqdm import tqdm
import museval
from collections import defaultdict

# =========================
# Configuration
# =========================

# RELATIVE PATHS: Defaulting to a folder named 'musdb18_hq' in your project root
# If your dataset is elsewhere, you can just change this one line.
MUSDB_ROOT = Path("./musdb18_test_samples") 
SEPARATED_ROOT = Path("./separated_musdb")
EVAL_ROOT = Path("./evaluated_musdb")
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

VOCALS_DIR = SEPARATED_ROOT / "vocals"
INSTR_DIR = SEPARATED_ROOT / "instrumentals"

OUTPUT_JSON = EVAL_ROOT / "museval_results.json"
OUTPUT_MEAN_JSON = EVAL_ROOT / "museval_means.json"
TARGET_SR = 44100

# =========================
# Helpers
# =========================

def stack_sources(vocals, accompaniment):
    # Museval expects shape: (n_sources, n_samples, n_channels)
    return np.stack([vocals, accompaniment], axis=0)

def load_audio(path: Path):
    audio, sr = sf.read(path, always_2d=True)
    if sr != TARGET_SR:
        raise ValueError(f"Sample rate mismatch: {path} ({sr} != {TARGET_SR})")
    return audio

def load_ground_truth(track_name: str):
    track_dir = MUSDB_ROOT / track_name
    v_path = track_dir / "vocals.wav"
    mix_path = track_dir / "mixture.wav"

    if not v_path.exists() or not mix_path.exists():
        # Fallback check: some versions use 'accompaniment.wav' directly
        acc_path = track_dir / "accompaniment.wav"
        if v_path.exists() and acc_path.exists():
            return load_audio(v_path), load_audio(acc_path)
        raise FileNotFoundError(f"Missing GT for {track_name} in {track_dir}")

    vocals = load_audio(v_path)
    mixture = load_audio(mix_path)
    # Ensure they are same length for subtraction
    min_len = min(len(vocals), len(mixture))
    accompaniment = mixture[:min_len] - vocals[:min_len]
    return vocals[:min_len], accompaniment

def parse_estimate_name(filename: str):
    """
    Parses our naming convention: {Song}_{Tool}_{Model}_{Device}_vocals.wav
    """
    stem = filename.replace("_vocals.wav", "")
    parts = stem.split("_")
    
    if "Demucs" in parts:
        idx = parts.index("Demucs")
        song, tool, model, device = "_".join(parts[:idx]), "Demucs", parts[idx+1], parts[idx+2]
    elif "OpenUnmix" in parts:
        idx = parts.index("OpenUnmix")
        song, tool, model, device = "_".join(parts[:idx]), "OpenUnmix", parts[idx+1], parts[idx+2]
    elif "Spleeter" in parts:
        idx = parts.index("Spleeter")
        song, tool, model, device = "_".join(parts[:idx]), "Spleeter", "default", parts[idx+1]
    else:
        raise ValueError(f"Unknown format: {filename}")
        
    return song, f"{tool}_{model}_{device}"

def calculate_means(results):
    grouped = defaultdict(list)
    for r in results:
        key = (r["system"], r["target"])
        grouped[key].append(r)

    mean_results = []
    for (system, target), items in grouped.items():
        # Clean NaNs which often happen in silent sections of MUSDB
        mean_sdr = np.nanmedian([np.nanmedian(i["SDR"]) for i in items])
        mean_sir = np.nanmedian([np.nanmedian(i["SIR"]) for i in items])
        
        mean_results.append({
            "system": system,
            "target": target,
            "mean_SDR": round(float(mean_sdr), 3),
            "mean_SIR": round(float(mean_sir), 3),
            "count_tracks": len(items)
        })
    return mean_results

# =========================
# Main evaluation
# =========================

def main(num_tracks=None):
    results = []

    vocal_files = sorted(VOCALS_DIR.glob("*.wav"))
    songs_seen = set()

    print(f"[INFO] Found {len(vocal_files)} estimated vocal tracks")

    for vocal_path in tqdm(vocal_files, desc="Evaluating"):
        try:
            song, system = parse_estimate_name(vocal_path.name)

            if num_tracks and len(songs_seen) >= num_tracks:
                break

            instr_path = INSTR_DIR / vocal_path.name.replace("_vocals.wav", "_instrumental.wav")
            if not instr_path.exists():
                print(f"[SKIP] Missing instrumental estimate for {vocal_path.name}")
                continue

            # Load estimates
            est_vocals = load_audio(vocal_path)
            est_instr = load_audio(instr_path)
            estimates = stack_sources(est_vocals, est_instr)

            # Load ground truth
            gt_vocals, gt_instr = load_ground_truth(song)
            references = stack_sources(gt_vocals, gt_instr)

            # Evaluate using museval
            sdr, isr, sir, sar = museval.evaluate(
                references, estimates, win=TARGET_SR, hop=TARGET_SR
            )

            # Store results
            targets = ["vocals", "accompaniment"]
            for i, target in enumerate(targets):
                results.append({
                    "track": song,
                    "system": system,
                    "target": target,
                    "SDR": sdr[i].tolist(),
                    "SIR": sir[i].tolist(),
                    "SAR": sar[i].tolist(),
                    "ISR": isr[i].tolist(),
                })

            songs_seen.add(song)

        except Exception as e:
            print(f"[ERROR] {vocal_path.name}: {e}")

    # Save full results
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Full evaluation results written to {OUTPUT_JSON}")

    # Compute mean per song/system/target and save
    mean_results = calculate_means(results)
    with open(OUTPUT_MEAN_JSON, "w") as f:
        json.dump(mean_results, f, indent=2)
    print(f"[INFO] Mean evaluation results written to {OUTPUT_MEAN_JSON}")

# =========================
# Entry point
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tracks", type=int, default=None,
                        help="Limit number of MUSDB tracks")
    args = parser.parse_args()
    main(args.num_tracks)
