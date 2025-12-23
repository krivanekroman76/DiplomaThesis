import warnings
warnings.simplefilter('ignore')  # Hide unnecessary warnings
import argparse
import os
import shutil
import tempfile
from pathlib import Path
from tqdm import tqdm

# === Import your existing separators (same as separation_app.py) ===
from separators.spleeter_separator import SpleeterSeparator
from separators.demucs_separator import DemucsSeparator
from separators.openunmix_separator import OpenUnmixSeparator

# =========================
# Configuration
# =========================

MUSDB_ROOT = Path(r"C:\Users\kriva\MUSDB18\MUSDB18-Converted\test")
OUTPUT_ROOT = Path("separated_musdb")
VOCALS_DIR = OUTPUT_ROOT / "vocals"
INSTR_DIR = OUTPUT_ROOT / "instrumentals"
TEXT_DIR = OUTPUT_ROOT / "transcription"

VOCALS_DIR.mkdir(parents=True, exist_ok=True)
INSTR_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
DEMUSC_MODELS = ["mdx", "mdx_extra", "htdemucs"]
OPENUNMIX_MODELS = ["umxl", "umxhq", "umx"]

# =========================
# Helpers
# =========================

def find_musdb_tracks(root: Path):
    tracks = []
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

def get_output_paths(tool_name, song_name, model_name=None):
    suffix = f"_{tool_name}"
    if model_name:
        suffix += f"_{model_name}"
    vocals_path = VOCALS_DIR / f"{song_name}{suffix}_vocals.wav"
    instr_path = INSTR_DIR / f"{song_name}{suffix}_instrumental.wav"
    return vocals_path, instr_path

# =========================
# Main separation logic
# =========================

def main(num_tracks=None):

    tracks = find_musdb_tracks(MUSDB_ROOT)

    if num_tracks is not None:
        tracks = tracks[:num_tracks]

    print(f"[INFO] Found {len(tracks)} MUSDB tracks")

    # --- Initialize separators ONCE ---
    spleeter = SpleeterSeparator()
    demucs = DemucsSeparator()
    openunmix = OpenUnmixSeparator()

    for song_name, mixture_path in tqdm(tracks, desc="Processing tracks"):
        print("\n" + "=" * 80)
        print(f"[TRACK] {song_name}")
        spleeter_input = prepare_named_input(mixture_path, song_name)
        print(f"[DEBUG] Original mixture: {mixture_path}")
        print(f"[DEBUG] Spleeter input  : {spleeter_input}")

        # -------------------------
        # Spleeter
        # -------------------------
        vocals_path, instr_path = get_output_paths("Spleeter", song_name)
        if vocals_path.exists() and instr_path.exists():
            print(f"[SKIP] Spleeter output already exists. Skipping separation.")
        else:
            try:
                success, _, _, _ = spleeter.separate(
                    input_path=str(spleeter_input),
                    song_name=song_name,
                    vocals_folder=str(VOCALS_DIR),
                    instr_folder=str(INSTR_DIR),
                    trans_folder=str(TEXT_DIR),
                    fmt="wav",
                    sr=44100,
                    do_transcribe=False,
                    progress_callback=None
                )
                if success:
                    print(f"[OK] Spleeter separation finished: {vocals_path}, {instr_path}")
                else:
                    print("[ERROR] Spleeter failed")
            except Exception as e:
                print(f"[ERROR] Spleeter exception: {e}")

        # -------------------------
        # Demucs
        # -------------------------
        for model in DEMUSC_MODELS:
            vocals_dest, instr_dest = get_output_paths("Demucs", song_name, model)
            if vocals_dest.exists() and instr_dest.exists():
                print(f"[SKIP] Demucs ({model}) output already exists. Skipping separation.")
                continue
            try:
                success, _, _, _ = demucs.separate(
                    input_path=str(mixture_path),
                    song_name=song_name,
                    vocals_folder=str(VOCALS_DIR),
                    instr_folder=str(INSTR_DIR),
                    trans_folder=str(TEXT_DIR),
                    model=model,
                    fmt="wav",
                    sr=44100,
                    do_transcribe=False,
                    progress_callback=None
                )
                if success:
                    print(f"[OK] Demucs ({model}) separation finished")
                else:
                    print(f"[ERROR] Demucs ({model}) failed")
            except Exception as e:
                print(f"[ERROR] Demucs ({model}) exception: {e}")

        # -------------------------
        # OpenUnmix
        # -------------------------
        for model in OPENUNMIX_MODELS:
            vocals_path, instr_path = get_output_paths("OpenUnmix", song_name, model)
            if vocals_path.exists() and instr_path.exists():
                print(f"[SKIP] OpenUnmix ({model}) output already exists. Skipping separation.")
            else:
                try:
                    success, _, _, _ = openunmix.separate(
                        input_path=str(mixture_path),
                        song_name=song_name,
                        vocals_folder=str(VOCALS_DIR),
                        instr_folder=str(INSTR_DIR),
                        trans_folder=str(TEXT_DIR),
                        model=model,
                        fmt="wav",
                        sr=44100,
                        do_transcribe=False,
                        progress_callback=None
                    )
                    if success:
                        print(f"[OK] OpenUnmix ({model}) separation finished: {vocals_path}, {instr_path}")
                    else:
                        print(f"[ERROR] OpenUnmix ({model}) failed")
                except Exception as e:
                    print(f"[ERROR] OpenUnmix ({model}) exception: {e}")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_tracks", type=int, default=None,
                        help="Limit number of MUSDB tracks")
    args = parser.parse_args()

    main(args.num_tracks)
