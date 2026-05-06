import os
import sys
import time
import json
import logging
from pathlib import Path
from tqdm import tqdm

# Ensure the repository root is on sys.path when executed from Evaluation/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# === Import your custom classes ===
from separators.whisper_transcription import WhisperTranscription
from separators.wav2vec2_transcription import Wav2Vec2Transcription
from separators.vosk_transcription import VoskTranscription

# --- Configuration ---
EVAL_ROOT = Path(__file__).resolve().parent
MUNI_BASE = EVAL_ROOT / "dataset_muni"
RAW_DIR = MUNI_BASE / "audio"
SEP_DIR = MUNI_BASE / "separated"
OUT_DIR = EVAL_ROOT / "transcriptions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Define the sources we are testing
DATA_SOURCES = {
    "Raw": RAW_DIR,
    "Spleeter": SEP_DIR / "spleeter",
    "Demucs": SEP_DIR / "demucs",
    "OpenUnmix": SEP_DIR / "openunmix"
}

# Define the models to use for each tool
# (Ensure these model names match your local folders/HuggingFace IDs)
TOOLS_TO_TEST = {
    "Whisper": {
        "class": WhisperTranscription(),
        "model_name": "medium",
        "params": {"language": "en"}
    },
    "Wav2Vec2": {
        "class": Wav2Vec2Transcription(),
        "model_name": "facebook/wav2vec2-large-960h-lv60-self",
        "params": {}
    },
    "Vosk": {
        "class": VoskTranscription(),
        "model_name": "vosk-model-en-us-0.22-lfm", # Ensure this folder exists in your Models/vosk/
        "params": {"use_diarization": False}
    }
}

def transcribe_all():
    results_log = []

    # Outer loop: The Tools (Whisper, Vosk, etc.)
    for tool_name, tool_cfg in TOOLS_TO_TEST.items():
        print(f"\n{'='*30}\n[TOOL] Switching to {tool_name}\n{'='*30}")
        transcriber = tool_cfg["class"]
        model_name = tool_cfg["model_name"]

        # Middle loop: The Sources (Raw vs. Separated)
        for source_label, source_path in DATA_SOURCES.items():
            if not source_path.exists():
                print(f"[SKIP] Source path not found: {source_path}")
                continue

            audio_files = list(source_path.rglob("*.wav")) + list(source_path.rglob("*.mp3"))
            print(f"\n[SOURCE] {source_label} ({len(audio_files)} files)")

            # Create output directory for this combination
            save_folder = OUT_DIR / tool_name / source_label
            save_folder.mkdir(parents=True, exist_ok=True)

            # Inner loop: The Audio Files
            for audio_path in tqdm(audio_files, desc=f"{tool_name} on {source_label}"):
                # Use a clean stem for the filename
                song_id = audio_path.stem
                output_txt_path = save_folder / f"{song_id}.txt"

                # Transcription Execution & Timing
                start_time = time.time()
                try:
                    # Dynamically call the transcribe method
                    # Most of your classes follow: transcribe(audio_path, output_path, model_name, **kwargs)
                    success, _ = transcriber.transcribe(
                        audio_path=str(audio_path),
                        output_path=str(output_txt_path),
                        model_name=model_name,
                        device_choice="Auto",
                        **tool_cfg["params"]
                    )
                    elapsed = time.time() - start_time

                    if success:
                        # Read result to get word count for stats
                        with open(output_txt_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            word_count = len(content.split())

                        results_log.append({
                            "song": song_id,
                            "source_tool": source_label,
                            "transcription_tool": tool_name,
                            "model_used": model_name,
                            "time_seconds": round(elapsed, 2),
                            "word_count": word_count,
                            "status": "success"
                        })
                    else:
                        results_log.append({"song": song_id, "source_tool": source_label, "status": "failed"})

                except Exception as e:
                    print(f"[ERROR] Failed {song_id} with {tool_name}: {e}")
                    results_log.append({"song": song_id, "source_tool": source_label, "status": f"error: {str(e)}"})

    # Save final metadata for evaluate_transcription.py
    stats_path = OUT_DIR / "transcription_metadata.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(results_log, f, indent=4)
    
    print(f"\n[COMPLETE] All transcriptions finished. Stats saved to {stats_path}")

if __name__ == "__main__":
    transcribe_all()