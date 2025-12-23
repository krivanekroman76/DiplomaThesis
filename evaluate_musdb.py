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

MUSDB_ROOT = Path(r"C:\Users\kriva\MUSDB18\MUSDB18-Converted\test")
SEPARATED_ROOT = Path("separated_musdb")
EVAL_ROOT = Path("evaluated_musdb")
EVAL_ROOT.mkdir(parents=True, exist_ok=True)  # Create folder if not exists

VOCALS_DIR = SEPARATED_ROOT / "vocals"
INSTR_DIR = SEPARATED_ROOT / "instrumentals"

OUTPUT_JSON = EVAL_ROOT / "museval_results.json"
OUTPUT_MEAN_JSON = EVAL_ROOT / "museval_means.json"
TARGET_SR = 44100

# =========================
# Helpers
# =========================

def load_audio(path: Path):
    audio, sr = sf.read(path, always_2d=True)
    if sr != TARGET_SR:
        raise ValueError(f"Sample rate mismatch: {path} ({sr} != {TARGET_SR})")
    return audio

def load_ground_truth(track_name: str):
    track_dir = MUSDB_ROOT / track_name
    vocals_path = track_dir / "vocals.wav"
    acc_path = track_dir / "accompaniment.wav"

    if not vocals_path.exists() or not acc_path.exists():
        raise FileNotFoundError(f"Missing ground truth for {track_name}")

    vocals = load_audio(vocals_path)
    accompaniment = load_audio(acc_path)
    return vocals, accompaniment

def parse_estimate_name(filename: str):
    stem = filename.replace("_vocals.wav", "")
    if "_Demucs_" in stem:
        song, model = stem.rsplit("_Demucs_", 1)
        system = f"Demucs_{model}"
    elif "_OpenUnmix_" in stem:
        song, model = stem.rsplit("_OpenUnmix_", 1)
        system = f"OpenUnmix_{model}"
    elif stem.endswith("_Spleeter"):
        song = stem.replace("_Spleeter", "")
        system = "Spleeter"
    else:
        raise ValueError(f"Unrecognized filename format: {filename}")
    return song, system

def stack_sources(vocals, accompaniment):
    return np.stack([vocals, accompaniment], axis=0)

def calculate_means(results):
    grouped = defaultdict(list)
    for r in results:
        key = (r["track"], r["system"], r["target"])
        grouped[key].append(r)

    mean_results = []
    for (track, system, target), items in grouped.items():
        mean_sdr = np.mean([np.mean(i["SDR"]) for i in items])
        mean_sir = np.mean([np.mean(i["SIR"]) for i in items])
        mean_sar = np.mean([np.mean(i["SAR"]) for i in items])
        mean_isr = np.mean([np.mean(i["ISR"]) for i in items])
        mean_results.append({
            "track": track,
            "system": system,
            "target": target,
            "mean_SDR": mean_sdr,
            "mean_SIR": mean_sir,
            "mean_SAR": mean_sar,
            "mean_ISR": mean_isr
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
