"""
================================================================================
TRANSCRIPTION EVALUATION Pipeline
================================================================================
AUTHOR: Bc. Roman Křivánek
REQUIRED: A dataset folder containing audio files and optionally .txt lyrics.

USAGE GUIDE:

1. FETCH LYRICS (Genius.com API):
    python Evaluation/evaluate_transcription.py --fetch_lyrics --genius_token YOUR_TOKEN # there might be a default token, but it's recommended to use your own for reliability
    NOTE: Lyrics are saved as .txt files. You can manually verify or edit these 
          files before running the evaluation to ensure "Ground Truth" accuracy.

2. SEPARATE & EVALUATE (Full Pipeline):
    python Evaluation/evaluate_transcription.py --run_separation
        (Uses default htdemucs)
    python Evaluation/evaluate_transcription.py --run_separation spleeter-2stems --num_tracks 5
        (Runs separation using Spleeter on only the first 5 tracks)
    NOTE: without --run_separation, the script will skip directly to transcription evaluation using existing separated vocals in the specified separated_path.

3. TARGETED EVALUATION (Specific Tools/Models):
    python Evaluation/evaluate_transcription.py --tools Whisper --models medium
    python Evaluation/evaluate_transcription.py --tools Wav2Vec2 Vosk --num_tracks 10
    
    NOTE: Default models for each tool are defined in the 'DEFAULT_CONFIG' 
dictionary at the top of this script. Edit that section to change defaults.

4. MULTILINGUAL EVALUATION:
    python Evaluation/evaluate_transcription.py --whisper_langs fr --tools Whisper
        (Tells Whisper to attempt forced French transcription)

5. CLEAN RUN (Overwrite cached .txt and CSV results):
    python Evaluation/evaluate_transcription.py --clean_run
    NOTE: Transcription will be redone, and all previous results will be overwritten. If added --run_separation it will also be redone.

FLAGS SUMMARY:
--device [Auto|cuda|cpu] : Hardware acceleration (Default: Auto) Cuda means GPU
--model_path [Path]      : Local directory where model weights are stored or downloaded to (Default: ./Models)
--clean_run              : Deletes existing files and forces a fresh run of separation and/or transcription.
--skip_transcription     : Runs only separation/lyrics fetching phases
================================================================================
"""
DEFAULT_GENIUS_TOKEN = "q5ySfDNJOqsEoKB6H7ZE7iBxTXmT_JD7kKfJCwLJFE0FzOT6pIx-HNVaPnrgVL2a"

# --- 1. USER CONFIGURATION (DEFAULT MODELS) ---
# Point users here to change what runs by default
DEFAULT_CONFIG = {
    "Whisper": ["small"],
    "Vosk": ["default"],
    "Wav2Vec2": ["facebook/wav2vec2-base-960h"],
    "Separation": "htdemucs"
}

import os
import re
import sys
import time
import json
import argparse
import platform
import shutil
import logging
import gc
import unicodedata
import zipfile
import io
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import jiwer
import lyricsgenius
import cpuinfo
from tinytag import TinyTag

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- SECURE IMPORTS & REGISTRY ---
# We use a dictionary to store classes to avoid "Unbound" errors in static analysis
STRATEGY_REGISTRY: Dict[str, Any] = {"separators": {}, "transcribers": {}}

try:
    from separators.spleeter_separator import SpleeterSeparator
    STRATEGY_REGISTRY["separators"]["spleeter"] = SpleeterSeparator
    from separators.demucs_separator import DemucsSeparator
    STRATEGY_REGISTRY["separators"]["demucs"] = DemucsSeparator
    from separators.openunmix_separator import OpenUnmixSeparator
    STRATEGY_REGISTRY["separators"]["openunmix"] = OpenUnmixSeparator
    
    from separators.whisper_transcription import WhisperTranscription
    STRATEGY_REGISTRY["transcribers"]["Whisper"] = WhisperTranscription
    from separators.vosk_transcription import VoskTranscription
    STRATEGY_REGISTRY["transcribers"]["Vosk"] = VoskTranscription
    from separators.wav2vec2_transcription import Wav2Vec2Transcription
    STRATEGY_REGISTRY["transcribers"]["Wav2Vec2"] = Wav2Vec2Transcription
except ImportError as e:
    logger.warning(f"Some modules could not be imported. Specific tools will be unavailable: {e}")

class TranscriptionEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.dataset_dir = Path(args.dataset_path)
        self.lyrics_dir = self.dataset_dir / "lyrics"
        self.separated_dir = Path(args.separated_path)
        self.results_dir = Path("Evaluation/transcription_results")
        self.txt_outputs_dir = self.results_dir / "txt_outputs"
        
        for p in [self.separated_dir, self.results_dir, self.txt_outputs_dir]:
            p.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.results_dir / "raw_transcription_metrics.csv"
        self.json_path = self.results_dir / "summary_transcription.json"
        self.devices_to_test = self._resolve_devices(args.device)

    def _resolve_devices(self, device_arg: str) -> List[str]:
        device_arg = device_arg.lower()
        if device_arg == "auto":
            return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
        return ["cuda"] if (device_arg in ["gpu", "cuda"] and torch.cuda.is_available()) else ["cpu"]

    def _get_cpu_name(self) -> str:
        try:
            return cpuinfo.get_cpu_info().get('brand_raw', 'Unknown CPU')
        except Exception:
            return platform.processor()

    def _parse_transcription_text(self, raw_output: str) -> str:
        if "Transcription (Model:" in raw_output and "Timestamps:" in raw_output:
            parts = raw_output.split("\n\nTimestamps:")[0]
            content = parts.split("):\n", 1)[-1]
            return content.strip()
        return raw_output.strip()

    def _get_ground_truth_lyrics(self, track_name: str) -> str:
        lyrics_path = self.lyrics_dir / f"{track_name}.txt"
        if lyrics_path.exists():
            return lyrics_path.read_text(encoding="utf-8").strip()
        return ""

    def _clean_genius_lyrics(self, raw_lyrics: str) -> str:
        lines = raw_lyrics.split('\n')
        if lines and ("Lyrics" in lines[0] or "Translations" in lines[0]):
            lines = lines[1:]
        text = "\n".join(lines)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\d*Embed$', '', text)
        
        cleaned_lines = []
        skip_keywords = ["MAKING OF", "TRADUCTION", "TRANSLATIONS", "CONTRIBUTORS", "READ MORE"]
        for line in text.split('\n'):
            line_s = line.strip()
            if not line_s or any(k in line_s.upper() for k in skip_keywords):
                continue
            if line_s.startswith(("On the ", "Cette chanson")) and len(line_s) > 35:
                continue
            cleaned_lines.append(line_s)
            
        return re.sub(r'\n{3,}', '\n\n', "\n".join(cleaned_lines)).strip()

    def robust_clean(self, text: str) -> str:
        text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = text.lower().replace("'", " ").replace("-", " ")
        return re.sub(r'[^\w\s]', '', text).strip()

    def fetch_and_save_lyrics(self):
        token = self.args.genius_token if self.args.genius_token else DEFAULT_GENIUS_TOKEN
        if not token:
            logger.error("No Genius API token provided.")
            return

        if self.lyrics_dir.exists():
            shutil.rmtree(self.lyrics_dir)
        self.lyrics_dir.mkdir(parents=True, exist_ok=True)

        genius = lyricsgenius.Genius(token, timeout=15, retries=3, verbose=False)
        audio_files = list(self.dataset_dir.glob("**/*.mp3")) + list(self.dataset_dir.glob("**/*.wav"))
        
        for i, track_path in enumerate(audio_files, 1):
            track_name = track_path.stem
            print(f"[{i}/{len(audio_files)}] Fetching lyrics for: {track_name}")
            try:
                tag = TinyTag.get(track_path)
                artist = (tag.artist or "Unknown").split(',')[0].strip()
                title = re.sub(r'\(feat.*?\)', '', tag.title or track_name, flags=re.I).strip()
                
                song = genius.search_song(title, artist)
                if song and song.lyrics:
                    clean = self._clean_genius_lyrics(song.lyrics)
                    (self.lyrics_dir / f"{track_name}.txt").write_text(clean, encoding="utf-8")
            except Exception as e:
                print(f"  [!] Error: {e}")
            time.sleep(0.5)

    def separate_vocals(self):
        if not self.args.run_separation: return

        sep_input = self.args.run_separation.lower()
        tool_key, model_name = sep_input.split("-", 1) if "-" in sep_input else (sep_input, DEFAULT_CONFIG["Separation"])
        sep_class = STRATEGY_REGISTRY["separators"].get(tool_key)
        
        if not sep_class: return

        os.environ["TORCH_HOME"] = str(Path(self.args.model_path).absolute())
        separator = sep_class()
        
        audio_files = list(self.dataset_dir.glob("**/*.mp3")) + list(self.dataset_dir.glob("**/*.wav"))
        if self.args.num_tracks: audio_files = audio_files[:self.args.num_tracks]

        for track_path in audio_files:
            final_vocal_path = self.separated_dir / f"{track_path.stem}.wav"
            
            if final_vocal_path.exists() and self.args.clean_run:
                final_vocal_path.unlink()
            
            if final_vocal_path.exists(): continue

            temp_out_dir = self.separated_dir / f"temp_{track_path.stem}"
            temp_out_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                kwargs = {
                    "input_path": str(track_path), "song_name": track_path.stem,
                    "vocals_folder": str(temp_out_dir), "instr_folder": str(temp_out_dir),
                    "device_choice": self.devices_to_test[0]
                }
                if tool_key in ["demucs", "openunmix"]: kwargs["model_name"] = model_name
                
                result = separator.separate(**kwargs)
                if result and result[0] and result[1]:
                    src = temp_out_dir / str(result[1])
                    if src.exists(): shutil.move(str(src), str(final_vocal_path))
                shutil.rmtree(temp_out_dir, ignore_errors=True)
                print(f" [+] Separated: {track_path.stem}")
            except Exception as e:
                print(f" [!] Error: {e}")

    def get_detailed_metrics(self, ref: str, est: str) -> Dict[str, Any]:
        """Compatible with JiWER 2.5.1"""
        measures = jiwer.compute_measures(ref, est)
        # Manually calculate CER as JiWER 2.x doesn't have a direct compute_measures for chars
        cer = jiwer.cer(ref, est)
        
        return {
            "wer": measures["wer"],
            "cer": cer,
            "w_sub": measures["substitutions"],
            "w_del": measures["deletions"],
            "w_ins": measures["insertions"]
        }

    def run_evaluation_pipeline(self):
        all_metrics: List[Any] = []
        transcriber: Any = None
        device: str = "cpu"

        if self.csv_path.exists() and not self.args.clean_run:
            try:
                all_metrics = pd.read_csv(self.csv_path).to_dict('records')
            except:
                all_metrics = []
        elif self.args.clean_run and self.csv_path.exists():
            self.csv_path.unlink()

        test_samples = list(self.separated_dir.glob("*.wav"))
        if self.args.num_tracks: test_samples = test_samples[:self.args.num_tracks]

        tools_to_run = self.args.tools or [k for k in DEFAULT_CONFIG.keys() if k != "Separation"]

        for tool_name in tools_to_run:
            transcriber_class = STRATEGY_REGISTRY["transcribers"].get(tool_name)
            if not transcriber_class: continue
            models = self.args.models or DEFAULT_CONFIG.get(tool_name, ["small"])

            for model_name in models:
                for device in self.devices_to_test:
                    langs = self.args.whisper_langs if tool_name == "Whisper" else [None]
                    transcriber = transcriber_class(custom_models_dir=self.args.model_path)

                    for lang in langs:
                        print(f"\n--- {tool_name} ({model_name}) | {device} | Lang: {lang} ---")
                        for track_path in test_samples:
                            wer, cer, rtf = 1.0, 1.0, 0.0
                            lang_tag = f"_{lang}" if lang else ""
                            cache_file = self.txt_outputs_dir / f"{tool_name}_{str(model_name).replace('/','_')}_{device}{lang_tag}_{track_path.stem}.txt"

                            try:
                                duration = librosa.get_duration(path=str(track_path)) # type: ignore
                            except:
                                duration = 1.0

                            if cache_file.exists() and not self.args.clean_run:
                                raw_out = cache_file.read_text(encoding="utf-8")
                            else:
                                start_t = time.time()
                                success, _ = transcriber.transcribe(audio_path=str(track_path), output_path=str(cache_file), model_name=model_name, device_choice=device, **({"language": lang} if lang else {}))
                                rtf = (time.time() - start_t) / duration if duration > 0 else 0
                                raw_out = cache_file.read_text(encoding="utf-8") if success else ""

                            if raw_out.strip():
                                ref_p = self.lyrics_dir / f"{track_path.stem}.txt"
                                ref_raw = ref_p.read_text(encoding="utf-8") if ref_p.exists() else ""
                                ref, est = self.robust_clean(ref_raw), self.robust_clean(raw_out)
                                if ref and est:
                                    m = self.get_detailed_metrics(ref, est)
                                    wer, cer = m["wer"], m["cer"]

                            row = {"tool": tool_name, "model": model_name, "lang": lang or "N/A", "device": device, "song": track_path.stem, "WER": wer, "CER": cer, "rtf": rtf}
                            all_metrics = [m for m in all_metrics if not (m.get('tool')==tool_name and m.get('model')==model_name and m.get('song')==track_path.stem and m.get('lang')==(lang or "N/A"))]
                            all_metrics.append(row)
                            pd.DataFrame(all_metrics).to_csv(self.csv_path, index=False)
                            print(f" -> {track_path.stem}: WER {wer*100:.1f}%")

                    if transcriber: del transcriber
                    gc.collect()
                    if device == "cuda": torch.cuda.empty_cache()
                    
    def generate_summaries(self):
        if not self.csv_path.exists(): return
        df = pd.read_csv(self.csv_path)
        summary = df.groupby(['tool', 'model']).agg({'WER': 'mean', 'CER': 'mean', 'rtf': 'mean'}).reset_index()
        summary.to_json(self.json_path, orient='records', indent=4)
        print("\n--- EVALUATION COMPLETE ---")
        print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="Evaluation/dataset_muni")
    parser.add_argument("--separated_path", type=str, default="Evaluation/separated_muni")
    parser.add_argument("--device", type=str, default="Auto")
    parser.add_argument("--num_tracks", type=int, default=None)
    parser.add_argument("--clean_run", action="store_true")
    parser.add_argument("--model_path", type=str, default="./Models")
    parser.add_argument("--run_separation", nargs="?", const="demucs-htdemucs", default=None)
    parser.add_argument("--tools", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--whisper_langs", nargs="+", default=["auto", "fr"], help="Languages to force Whisper to use. Defaults to both 'auto' and 'fr'.")
    parser.add_argument("--skip_transcription", action="store_true")
    parser.add_argument("--fetch_lyrics", action="store_true")
    parser.add_argument("--genius_token", type=str, default="")
    
    args = parser.parse_args()
    evaluator = TranscriptionEvaluator(args)
    
    if args.fetch_lyrics:
        evaluator.fetch_and_save_lyrics()
    else:
        print("\n[*] Skipping lyrics fetching phase (--fetch_lyrics flag not passed). Assuming lyrics are already in place.")
    if args.run_separation:
        evaluator.separate_vocals()
    else:
        print("\n[*] Skipping separation phase (--run_separation flag not passed). Assuming vocals are already in the separated directory.")
    if not args.skip_transcription:
        evaluator.run_evaluation_pipeline()
        evaluator.generate_summaries()
    else:
        print("\n[*] Skipping transcription evaluation phase (--skip_transcription flag passed).")
