"""
================================================================================
MUNI DATASET PREPARATION SCRIPT
================================================================================
DESCRIPTION:
Walks through the MUNI dataset (nested folders), separates vocals using 
Spleeter, Demucs (htdemucs), and OpenUnmix (umxl), and saves them for transcription.

USAGE:
1. Ensure your separators (spleeter_separator.py, etc.) are in the same directory.
2. Place raw audio in: ./dataset_muni/audio/{Artist}/{Album}/*
3. Run: python prepare_muni.py
================================================================================
"""

import os
import gc
import torch
from pathlib import Path
from tqdm import tqdm

# Import your existing separator classes
from separators.spleeter_separator import SpleeterSeparator
from separators.demucs_separator import DemucsSeparator
from separators.openunmix_separator import OpenUnmixSeparator

# --- Configuration ---
RAW_AUDIO_ROOT = Path("./dataset_muni/audio")
SEPARATED_ROOT = Path("./dataset_muni/separated")
DEVICE = "GPU" # Set to "CPU" if no GPU is available
FORCE_REPROCESS = False

# Models to use for the "High-Middle-Low" evaluation
MODELS = {
    "Spleeter",
    "Demucs": "htdemucs", # Recommended SOTA
    "OpenUnmix": "umxl"
}

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    # Initialize tools
    spleeter = SpleeterSeparator()
    demucs = DemucsSeparator()
    unmix = OpenUnmixSeparator()

    # Find all audio files (assuming .wav or .mp3)
    audio_files = list(RAW_AUDIO_ROOT.rglob("*.wav")) + list(RAW_AUDIO_ROOT.rglob("*.mp3"))
    print(f"[INFO] Found {len(audio_files)} tracks in MUNI dataset.")

    for audio_path in tqdm(audio_files, desc="Processing MUNI"):
        # Create a unique name that preserves folder structure
        # e.g., "Black M_AlbumName_TrackName"
        relative_path = audio_path.relative_to(RAW_AUDIO_ROOT)
        song_id = "_".join(relative_path.with_suffix('').parts)
        
        # 1. Spleeter
        sep_dir = SEPARATED_ROOT / "spleeter"
        v_dest = sep_dir / f"{song_id}_Spleeter_vocals.wav"
        if FORCE_REPROCESS or not v_dest.exists():
            spleeter.separate(str(audio_path), song_id, str(sep_dir), "./temp", device_choice=DEVICE)
            free_memory()

        # 2. Demucs (htdemucs)
        sep_dir = SEPARATED_ROOT / "demucs"
        v_dest = sep_dir / f"{song_id}_Demucs_htdemucs_vocals.wav"
        if FORCE_REPROCESS or not v_dest.exists():
            demucs.separate(str(audio_path), song_id, str(sep_dir), "./temp", 
                            model=MODELS["Demucs"], device_choice=DEVICE)
            free_memory()

        # 3. OpenUnmix (umxl)
        sep_dir = SEPARATED_ROOT / "openunmix"
        v_dest = sep_dir / f"{song_id}_OpenUnmix_umxl_vocals.wav"
        if FORCE_REPROCESS or not v_dest.exists():
            unmix.separate(str(audio_path), song_id, str(sep_dir), "./temp", 
                           model=MODELS["OpenUnmix"], device_choice=DEVICE)
            free_memory()

    print("\n[SUCCESS] MUNI Dataset separated. Ready for transcription.")

if __name__ == "__main__":
    main()