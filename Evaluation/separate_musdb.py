"""
================================================================================
MUSDB18 BENCHMARKING SCRIPT: REPRODUCIBILITY GUIDE
================================================================================

DESCRIPTION:
This script performs a standardized evaluation of audio separation models 
(Spleeter, Demucs, OpenUnmix) across different hardware backends (CPU/GPU).
It measures execution time and saves the output stems for quality analysis.

HOW TO REPRODUCE THE RESULTS:
0. Download the MUSDB18-HQ dataset from: https://zenodo.org/records/3338373. (22,7 GB in zip)
1. DATASET: Place the MUSDB18-HQ 'test' folder tracks into './musdb18_test_samples'.
   Each subfolder must contain a 'mixture.wav'.
2. ENVIRONMENT: Ensure all dependencies (torch, spleeter, demucs, openunmix) are 
   installed as per the project requirements.
3. EXECUTION:
   - To run the full benchmark (all tracks, CPU & GPU):
     python separate_musdb.py
   - To run a quick test (e.g., first 2 tracks):
     python separate_musdb.py --num_tracks 2
4. OUTPUTS:
   - Timings: Saved to './separated_musdb/separation_times.json'
   - Stems:   Saved to './separated_musdb/vocals' and '.../instrumentals'
     with the naming convention: {TrackName}_{Tool}_{Model}_{Device}_{Target}.wav

HARDWARE NOTE: 
SDR/SIR quality metrics are hardware-independent. Timing results are highly 
dependent on the specific CPU and GPU models used.
================================================================================
"""

import warnings
warnings.simplefilter('ignore')

import argparse
import os
import shutil
import tempfile
import time
import json
import gc
from pathlib import Path
from tqdm import tqdm

try:
    import torch
except ImportError:
    pass

# === Import your existing separators ===
from separators.spleeter_separator import SpleeterSeparator
from separators.demucs_separator import DemucsSeparator
from separators.openunmix_separator import OpenUnmixSeparator

# =========================
# Configuration
# =========================

MUSDB_ROOT = Path("./musdb18_test_samples")
OUTPUT_ROOT = Path("./separated_musdb")
VOCALS_DIR = OUTPUT_ROOT / "vocals"
INSTR_DIR = OUTPUT_ROOT / "instrumentals"
TIMES_JSON_PATH = OUTPUT_ROOT / "separation_times.json"

# REPRODUCIBILITY CONTROL: 
# Set to True to delete existing files and re-run everything.
# Set to False to skip files that are already in the output folders.
FORCE_REPROCESS = False 

DEMUCS_MODELS = ["mdx", "mdx_extra", "htdemucs"]
OPENUNMIX_MODELS = ["umxl", "umxhq", "umx"]
DEVICES_TO_TEST = ["GPU", "CPU"]

VOCALS_DIR.mkdir(parents=True, exist_ok=True)
INSTR_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Helpers
# =========================

def free_memory():
    gc.collect()
    if 'torch' in globals() and torch.cuda.is_available():
        torch.cuda.empty_cache()

def find_musdb_tracks(root: Path):
    tracks = []
    if not root.exists():
        print(f"[WARNING] Dataset path {root} not found!")
        return tracks
    for track_dir in sorted(root.iterdir()):
        mixture = track_dir / "mixture.wav"
        if track_dir.is_dir() and mixture.exists():
            tracks.append((track_dir.name, mixture))
    return tracks

def prepare_named_input(original_mixture: Path, song_name: str) -> Path:
    tmp_dir = Path(tempfile.mkdtemp())
    safe_name = song_name.replace("/", "_").replace("\\", "_")
    new_path = tmp_dir / f"{safe_name}.wav"
    shutil.copy(original_mixture, new_path)
    return new_path

def get_output_paths(tool_name, song_name, device, model_name=None):
    suffix = f"_{tool_name}"
    if model_name: suffix += f"_{model_name}"
    suffix += f"_{device}"
    
    vocals_path = VOCALS_DIR / f"{song_name}{suffix}_vocals.wav"
    instr_path = INSTR_DIR / f"{song_name}{suffix}_instrumental.wav"
    return vocals_path, instr_path

def save_time_to_json(song_name, tool, model, device, time_taken):
    """
    Checks for hardware availability and logs benchmark results.
    'device' is the requested device (GPU/CPU).
    """
    actual_hw = "CPU"
    try:
        import torch
        # If GPU was requested, check if it was actually available
        if device == "GPU":
            if torch.cuda.is_available():
                actual_hw = torch.cuda.get_device_name(0)
            else:
                actual_hw = "CPU (Fallback - CUDA not found)"
    except ImportError:
        actual_hw = "CPU (torch not installed)"

    data = []
    if TIMES_JSON_PATH.exists():
        with open(TIMES_JSON_PATH, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass
            
    data.append({
        "song": song_name,
        "tool": tool,
        "model": model,
        "requested_backend": device,
        "actual_hardware_used": actual_hw,
        "time_seconds": round(time_taken, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })
    
    with open(TIMES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# =========================
# Main logic
# =========================

def main(num_tracks=None):
    # --- HARDWARE CHECK ---
    gpu_available = False
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except:
        pass

    if "GPU" in DEVICES_TO_TEST and not gpu_available:
        print("[WARNING] GPU was requested but no CUDA-enabled GPU or drivers were found.")
        print("[ACTION] Removing 'GPU' from test list. Only 'CPU' will be benchmarked.")
        DEVICES_TO_TEST.remove("GPU")
    # ----------------------

    tracks = find_musdb_tracks(MUSDB_ROOT)
    if num_tracks is not None:
        tracks = tracks[:num_tracks]

    print(f"[INFO] Found {len(tracks)} tracks. Starting Benchmark.")

    # Initialize once
    spleeter = SpleeterSeparator()
    demucs = DemucsSeparator()
    openunmix = OpenUnmixSeparator()

    for song_name, mixture_path in tqdm(tracks, desc="Tracks"):
        # Spleeter needs a specific folder structure sometimes, 
        # so we use your helper
        spleeter_input = prepare_named_input(mixture_path, song_name)

        for device in DEVICES_TO_TEST:
            # Match the "Auto", "GPU", "CPU" logic in your modules
            target_hw = "GPU" if device == "GPU" else "CPU"

            # --- 1. Spleeter ---
            v_p, i_p = get_output_paths("Spleeter", song_name, device)
            if FORCE_REPROCESS or not v_p.exists():
                if v_p.exists(): v_p.unlink(); i_p.unlink()
                start = time.time()
                spleeter.separate(
                    input_path=str(spleeter_input), 
                    song_name=song_name, 
                    vocals_folder=str(VOCALS_DIR), 
                    instr_folder=str(INSTR_DIR), 
                    device_choice=target_hw  # Matches your module
                )
                save_time_to_json(song_name, "Spleeter", "2stems", device, time.time() - start)
                free_memory()

            # --- 2. Demucs ---
            for model in DEMUCS_MODELS:
                v_p, i_p = get_output_paths("Demucs", song_name, device, model)
                if FORCE_REPROCESS or not v_p.exists():
                    if v_p.exists(): v_p.unlink(); i_p.unlink()
                    start = time.time()
                    demucs.separate(
                        input_path=str(mixture_path), 
                        song_name=song_name, 
                        vocals_folder=str(VOCALS_DIR), 
                        instr_folder=str(INSTR_DIR), 
                        model=model, 
                        device_choice=target_hw, # Matches your module
                        shifts=1
                    )
                    save_time_to_json(song_name, "Demucs", model, device, time.time() - start)
                    free_memory()

            # --- 3. OpenUnmix ---
            for model in OPENUNMIX_MODELS:
                v_p, i_p = get_output_paths("OpenUnmix", song_name, device, model)
                if FORCE_REPROCESS or not v_p.exists():
                    if v_p.exists(): v_p.unlink(); i_p.unlink()
                    start = time.time()
                    openunmix.separate(
                        input_path=str(mixture_path), 
                        song_name=song_name, 
                        vocals_folder=str(VOCALS_DIR), 
                        instr_folder=str(INSTR_DIR), 
                        model=model, 
                        device_choice=target_hw # Matches your module
                    )
                    save_time_to_json(song_name, "OpenUnmix", model, device, time.time() - start)
                    free_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tracks", type=int, default=None)
    args = parser.parse_args()
    main(args.num_tracks)