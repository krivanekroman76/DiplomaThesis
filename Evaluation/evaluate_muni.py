import json
from pathlib import Path
import jiwer # pip install jiwer
import librosa

# Paths
GT_DIR = Path("./dataset_muni/ground_truth")
TRANS_DIR = Path("./transcriptions")
STATS_FILE = TRANS_DIR / "transcription_metadata.json"

def clean_text(text):
    # Professional normalization for French
    text = text.lower()
    # Remove punctuation that isn't essential for ASR
    for p in ".,!?;:": text = text.replace(p, "")
    return " ".join(text.split()) # Remove extra spaces

def run_evaluation():
    # Load the timing data from the transcription run
    with open(STATS_FILE, "r") as f:
        logs = json.load(f)
    
    final_table = []

    for entry in logs:
        song_id = entry["song"]
        tool = entry["transcription_tool"]
        source = entry["source_tool"]
        
        # 1. Load Hypothesis (AI Output)
        hyp_path = TRANS_DIR / tool / source / f"{song_id}.txt"
        if not hyp_path.exists(): continue
        hypothesis = clean_text(hyp_path.read_text(encoding="utf-8"))
        
        # 2. Load Reference (Ground Truth Lyrics)
        # Mapping: SongID usually contains artist_album_track
        # Make sure your GT files match the song_id
        ref_path = GT_DIR / f"{song_id}.txt" 
        if not ref_path.exists(): continue
        reference = clean_text(ref_path.read_text(encoding="utf-8"))

        # 3. Calculate Error Rates
        word_error = jiwer.wer(reference, hypothesis)
        char_error = jiwer.cer(reference, hypothesis)
        
        # 4. Calculate RTF (Real-Time Factor)
        # Assuming you have access to raw audio to get duration
        # Or you can store duration in the JSON during transcription
        # rtf = entry["time_seconds"] / audio_duration

        final_table.append({
            "Tool": tool,
            "Source": source,
            "WER": round(word_error * 100, 2), # Display as %
            "CER": round(char_error * 100, 2), # Display as %
            "RTF": entry.get("rtf", 0) # Use the RTF we calculated
        })
    
    # Save for LaTeX
    return final_table