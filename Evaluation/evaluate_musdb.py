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
import librosa

# =========================
# Configuration
# =========================

EVAL_ROOT = Path(__file__).resolve().parent
# RELATIVE PATHS: Evaluation data now lives inside the Evaluation folder
MUSDB_ROOT = EVAL_ROOT / "musdb18_test_samples"
SEPARATED_ROOT = EVAL_ROOT / "separated_musdb"
TIMES_JSON_PATH = SEPARATED_ROOT / "separation_times.json"
EVAL_ROOT.mkdir(parents=True, exist_ok=True)

VOCALS_DIR = SEPARATED_ROOT / "vocals"
INSTR_DIR = SEPARATED_ROOT / "instrumentals"

OUTPUT_JSON = EVAL_ROOT / "museval_results.json"
OUTPUT_MEAN_JSON = EVAL_ROOT / "museval_means.json"
OUTPUT_RTF_JSON = EVAL_ROOT / "rtf_results.json"
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

def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds."""
    try:
        duration = librosa.get_duration(filename=str(audio_path))
        return round(duration, 2)
    except Exception as e:
        print(f"[WARNING] Could not get duration for {audio_path}: {e}")
        return 0.0

def calculate_rtf(separation_time_seconds: float, audio_duration_seconds: float) -> float:
    """Calculate Real-Time Factor: separation_time / audio_duration."""
    if audio_duration_seconds <= 0:
        return 0.0
    return round(separation_time_seconds / audio_duration_seconds, 3)

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
    stem = filename.replace("_vocals.wav", "").replace("_vocals_1.wav", "")
    parts = stem.split("_")
    
    if "Demucs" in parts:
        idx = parts.index("Demucs")
        song, tool, model, device = "_".join(parts[:idx]), "Demucs", parts[idx+1], parts[idx+2]
    elif "OpenUnmix" in parts:
        idx = parts.index("OpenUnmix")
        song, tool, model, device = "_".join(parts[:idx]), "OpenUnmix", parts[idx+1], parts[idx+2]
    elif "Spleeter" in parts:
        idx = parts.index("Spleeter")
        song, tool, model, device = "_".join(parts[:idx]), "Spleeter", "2stems", parts[idx+1]
    else:
        raise ValueError(f"Unknown format: {filename}")
        
    return song, tool, model, device

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
            song, tool, model, device = parse_estimate_name(vocal_path.name)
            
            # ACCURACY METRICS: GPU-only to avoid duplicate evaluation
            # (algorithms are identical, only timing differs)
            if device != "GPU":
                continue
            
            system = f"{tool}_{model}"

            if num_tracks and len(songs_seen) >= num_tracks:
                break

            instr_path = INSTR_DIR / vocal_path.name.replace("_vocals.wav", "_instrumental.wav").replace("_vocals_1.wav", "_instrumental.wav")
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
    
    # === RTF CALCULATION FROM TIMING DATA ===
    rtf_results = []
    if TIMES_JSON_PATH.exists():
        with open(TIMES_JSON_PATH, "r") as f:
            try:
                timing_data = json.load(f)
                # Group by tool, model, requested_backend to compute aggregate RTF
                for entry in timing_data:
                    song = entry.get("song", "")
                    tool = entry.get("tool", "")
                    model = entry.get("model", "")
                    device = entry.get("requested_backend", "")
                    sep_time = entry.get("separation_time_seconds", 0.0)
                    audio_duration = entry.get("audio_duration_seconds", 0.0)
                    load_time = entry.get("model_load_time_seconds")
                    
                    rtf = calculate_rtf(sep_time, audio_duration)
                    
                    rtf_entry = {
                        "song": song,
                        "tool": tool,
                        "model": model,
                        "device": device,
                        "separation_time_seconds": sep_time,
                        "audio_duration_seconds": audio_duration,
                        "rtf": rtf
                    }
                    if load_time is not None:
                        rtf_entry["model_load_time_seconds"] = load_time
                    
                    rtf_results.append(rtf_entry)
                
                with open(OUTPUT_RTF_JSON, "w") as f:
                    json.dump(rtf_results, f, indent=2)
                print(f"[INFO] RTF results written to {OUTPUT_RTF_JSON}")
            except Exception as e:
                print(f"[WARNING] Could not process timing data: {e}")
    else:
        print(f"[WARNING] No timing data found at {TIMES_JSON_PATH}. Skipping RTF calculation.")

# =========================
# Entry point
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tracks", type=int, default=None,
                        help="Limit number of MUSDB tracks")
    args = parser.parse_args()
    main(args.num_tracks)
