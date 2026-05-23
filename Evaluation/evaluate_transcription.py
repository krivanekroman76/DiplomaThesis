"""
================================================================================
TRANSCRIPTION EVALUATION Pipeline (5-Tier, Alignment & Visual Summaries)
================================================================================
AUTHOR: Bc. Roman Křivánek
REQUIRED: A dataset folder containing audio files and optionally .txt lyrics.

USAGE GUIDE:

1. FETCH LYRICS (Genius.com API):
    python Evaluation/evaluate_transcription.py --fetch_lyrics --genius_token YOUR_TOKEN

2. SEPARATE & EVALUATE (Full Pipeline):
    python Evaluation/evaluate_transcription.py --run_separation

3. TARGETED EVALUATION WITH VISUAL ALIGNMENT (Specific Tools/Models):
    python Evaluation/evaluate_transcription.py --tools Whisper --models small --align_track_idx 0
================================================================================
"""
DEFAULT_GENIUS_TOKEN = "q5ySfDNJOqsEoKB6H7ZE7iBxTXmT_JD7kKfJCwLJFE0FzOT6pIx-HNVaPnrgVL2a"

# --- 1. USER CONFIGURATION (DEFAULT MODELS) ---
# (Assumes your backend uses Hugging Face transformers)
DEFAULT_CONFIG = {
    "Vosk": ["vosk-model-fr-0.22", "vosk-model-fr-0.6-linto-2.2.0"],
    "Wav2Vec2": [
        "jonatasgrosman/wav2vec2-large-xlsr-53-french", # The Competition Winner (Hugging Face speech recognition competition)
        "facebook/wav2vec2-large-xlsr-53-french"        # Official model from facebook
    ],
    "Whisper": [
        "openai/whisper-large-v3",                  # Official OpenAI benchmark
        "bofenghuang/whisper-large-v2-french",      # Massive 2,200h French fine-tune
        "bofenghuang/whisper-medium-french"         # Lighter, faster French fine-tune
    ],
    "Separation": ["htdemucs"] # demucs V4 model (Hybrid Transformer Demucs)
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
import difflib
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

# --- CONDITIONAL IMPORTS FOR PLOTTING (Pylance safe) ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    plt = None  # type: ignore
    sns = None  # type: ignore
    PLOTTING_AVAILABLE = False
    print("[!] 'matplotlib' or 'seaborn' not found. Graph generation will be skipped.")

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# --- SECURE IMPORTS & REGISTRY ---
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
        
        if args.clean_run and self.results_dir.exists():
            logger.info("🧹 [--clean_run] Wiping transcription results directory...")
            shutil.rmtree(self.results_dir, ignore_errors=True)
            
        if args.force_reseparate and self.separated_dir.exists():
            logger.info("🧹 [--force_reseparate] Wiping separated audio directory...")
            shutil.rmtree(self.separated_dir, ignore_errors=True)

        self.txt_outputs_dir = self.results_dir / "txt_outputs"
        self.plots_dir = self.results_dir / "plots"
        
        for p in [self.separated_dir, self.results_dir, self.txt_outputs_dir, self.plots_dir]:
            p.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.results_dir / "raw_transcription_metrics.csv"
        self.json_path = self.results_dir / "summary_transcription.json"
        self.latex_path = self.results_dir / "latex_table.tex"
        self.devices_to_test = self._resolve_devices(args.device)
        self.hardware_name = self._get_cuda_name() if "cuda" in self.devices_to_test else self._get_cpu_name()

    # --- 5 CLEANING TIERS ---
    def tier_1_minimalist(self, text: str) -> str:
        text = text.lower()
        return re.sub(r'\s+', ' ', text).strip()

    def tier_2_punctuation_strip(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s\'-]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tier_3_boundary_split(self, text: str) -> str:
        text = text.lower().replace("'", " ").replace("-", " ")
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tier_4_acoustic_robust(self, text: str) -> str:
        text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = text.lower().replace("'", " ").replace("-", " ")
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    def tier_5_aggressive_squash(self, text: str) -> str:
        text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = text.lower().replace("'", "").replace("-", "")
        text = re.sub(r'[^\w\s]', '', text)
        return re.sub(r'\s+', ' ', text).strip()

    # --- EXTRACTION METHODS ---
    def extract_continuous_text(self, raw_text: str) -> str:
        if "Transcription (Model:" in raw_text and "Timestamps:" in raw_text:
            parts = raw_text.split("\n\nTimestamps:")[0]
            content = parts.split("):\n", 1)[-1]
            return content.strip()
        return raw_text.strip()

    def extract_timestamped_text(self, raw_text: str) -> str:
        if "Timestamps:" in raw_text:
            timestamp_section = raw_text.split("Timestamps:")[1].strip()
            lines = timestamp_section.split('\n')
            extracted_lines = []
            for line in lines:
                cleaned_line = re.sub(r'^\d+\.\d+s\s*-\s*\d+\.\d+s:\s*', '', line.strip())
                if cleaned_line:
                    extracted_lines.append(cleaned_line)
            return " ".join(extracted_lines)
        return raw_text.strip()

    # --- ALIGNMENT VISUALIZATION ---
    def generate_alignment_visualization(self, ref_text: str, hyp_text: str, tool_info: str, chunk_size: int = 6):
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        
        matcher = difflib.SequenceMatcher(None, ref_words, hyp_words)
        aligned_ref, aligned_hyp, aligned_ops = [], [], []
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                for r, h in zip(ref_words[i1:i2], hyp_words[j1:j2]):
                    aligned_ref.append(r); aligned_hyp.append(h); aligned_ops.append(" ")
            elif tag == 'replace':
                r_chunk, h_chunk = ref_words[i1:i2], hyp_words[j1:j2]
                for r, h in zip(r_chunk, h_chunk):
                    aligned_ref.append(r); aligned_hyp.append(h); aligned_ops.append("SUB")
                if len(r_chunk) > len(h_chunk):
                    for r in r_chunk[len(h_chunk):]:
                        aligned_ref.append(r); aligned_hyp.append("*" * len(r)); aligned_ops.append("DEL")
                elif len(h_chunk) > len(r_chunk):
                    for h in h_chunk[len(r_chunk):]:
                        aligned_ref.append("*" * len(h)); aligned_hyp.append(h); aligned_ops.append("INS")
            elif tag == 'delete':
                for r in ref_words[i1:i2]:
                    aligned_ref.append(r); aligned_hyp.append("*" * len(r)); aligned_ops.append("DEL")
            elif tag == 'insert':
                for h in hyp_words[j1:j2]:
                    aligned_ref.append("*" * len(h)); aligned_hyp.append(h); aligned_ops.append("INS")

        print("\n" + "="*80)
        print(f" VISUAL ALIGNMENT: {tool_info}")
        print(" REF = Ground Truth Lyrics | HYP = Model Output (Tier 2 Applied)")
        print("="*80)

        for i in range(0, len(aligned_ops), chunk_size):
            r_chunk = aligned_ref[i:i+chunk_size]
            h_chunk = aligned_hyp[i:i+chunk_size]
            o_chunk = aligned_ops[i:i+chunk_size]
            
            widths = [max(len(r), len(h), len(o)) for r, h, o in zip(r_chunk, h_chunk, o_chunk)]
            
            ref_row = "REF: " + "  ".join(f"{word:<{widths[idx]}}" for idx, word in enumerate(r_chunk))
            hyp_row = "HYP: " + "  ".join(f"{word:<{widths[idx]}}" for idx, word in enumerate(h_chunk))
            op_row  = "OP:  " + "  ".join(f"{word:<{widths[idx]}}" for idx, word in enumerate(o_chunk))
            
            print(ref_row)
            print(hyp_row)
            print(op_row)
            print("-" * 80)

    # --- DEVICE & GENERAL UTILS ---
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

    def _get_cuda_name(self) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "N/A"

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
        
        # FIX: Safely extract the first model string if "Separation" is configured as a list
        default_sep = DEFAULT_CONFIG["Separation"]
        default_model = default_sep[0] if isinstance(default_sep, list) else default_sep

        # Now model_name is guaranteed to be a string type
        tool_key, model_name = sep_input.split("-", 1) if "-" in sep_input else (sep_input, default_model)
        sep_class = STRATEGY_REGISTRY["separators"].get(tool_key)
        
        if not sep_class: return

        os.environ["TORCH_HOME"] = str(Path(self.args.model_path).absolute())
        separator = sep_class()
        
        audio_files = list(self.dataset_dir.glob("**/*.mp3")) + list(self.dataset_dir.glob("**/*.wav"))
        if self.args.num_tracks: audio_files = audio_files[:self.args.num_tracks]

        if self.args.debug:
            logger.debug(f"[Separation Debug] Hardware target initialized: {self.hardware_name}")

        for track_path in audio_files:
            final_vocal_path = self.separated_dir / f"{track_path.stem}.wav"
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
                
                start_sep = time.time()
                result = separator.separate(**kwargs)
                elapsed_sep = time.time() - start_sep

                if self.args.debug:
                    logger.debug(f"  [Separation Time] Tool: {tool_key} | Track: {track_path.stem} | Time: {elapsed_sep:.2f}s")

                if result and result[0] and result[1]:
                    src = temp_out_dir / str(result[1])
                    if src.exists(): shutil.move(str(src), str(final_vocal_path))
                shutil.rmtree(temp_out_dir, ignore_errors=True)
                print(f" [+] Separated: {track_path.stem}")
            except Exception as e:
                print(f" [!] Error: {e}")

    def get_detailed_metrics(self, ref: str, est: str) -> Dict[str, Any]:
        if not ref or not est:
            return {"wer": 1.0, "cer": 1.0, "w_sub": 0, "w_del": 0, "w_ins": 0}
        measures = jiwer.compute_measures(ref, est)
        cer = jiwer.cer(ref, est)
        return {
            "wer": measures["wer"], "cer": cer,
            "w_sub": measures["substitutions"], "w_del": measures["deletions"], "w_ins": measures["insertions"]
        }

    def run_evaluation_pipeline(self):
        all_metrics: List[Any] = []
        transcriber: Any = None

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
        
        tiers = [
            self.tier_1_minimalist,
            self.tier_2_punctuation_strip,
            self.tier_3_boundary_split,
            self.tier_4_acoustic_robust,
            self.tier_5_aggressive_squash
        ]

        for tool_name in tools_to_run:
            transcriber_class = STRATEGY_REGISTRY["transcribers"].get(tool_name)
            if not transcriber_class: continue
            models = self.args.models or DEFAULT_CONFIG.get(tool_name, ["small"])

            for model_name in models:
                for device in self.devices_to_test:
                    langs = self.args.whisper_langs if tool_name == "Whisper" else [None]

                    import inspect
                    sig = inspect.signature(transcriber_class.__init__)
                    
                    if "custom_models_dir" in sig.parameters:
                        transcriber = transcriber_class(custom_models_dir=self.args.model_path)
                    elif "model_path" in sig.parameters:
                        transcriber = transcriber_class(model_path=self.args.model_path)
                    else:
                        transcriber = transcriber_class()

                    for lang in langs:
                        print(f"\n--- {tool_name} ({model_name}) | {device} | Lang: {lang} ---")
                        for track_idx, track_path in enumerate(test_samples):
                            rtf = 0.0
                            lang_tag = f"_{lang}" if lang else ""
                            cache_file = self.txt_outputs_dir / f"{tool_name}_{str(model_name).replace('/','_')}_{device}{lang_tag}_{track_path.stem}.txt"

                            try:
                                duration = sf.info(str(track_path)).duration
                            except Exception as e:
                                logger.error(f"  [!] Failed to get duration for {track_path.stem}: {e}")
                                duration = 1.0

                            if cache_file.exists() and not self.args.clean_run:
                                raw_out = cache_file.read_text(encoding="utf-8")
                                if self.args.debug: logger.debug(f"  [Cache Hit] Track: {track_path.stem}")
                            else:
                                start_t = time.time()
                                success, _ = transcriber.transcribe(
                                    audio_path=str(track_path), output_path=str(cache_file), 
                                    model_name=model_name, device_choice=device, 
                                    **({"language": lang} if lang else {})
                                )
                                elapsed_transcription = time.time() - start_t
                                rtf = elapsed_transcription / duration if duration > 0 else 0
                                raw_out = cache_file.read_text(encoding="utf-8") if success else ""

                            if not raw_out.strip(): continue

                            ref_p = self.lyrics_dir / f"{track_path.stem}.txt"
                            ref_raw = ref_p.read_text(encoding="utf-8") if ref_p.exists() else ""
                            
                            hyp_cont = self.extract_continuous_text(raw_out)
                            hyp_time = self.extract_timestamped_text(raw_out)
                            
                            is_unified = (hyp_cont == hyp_time)
                            text_variants = [("unified", hyp_cont)] if is_unified else [("continuous", hyp_cont), ("timestamped", hyp_time)]
                            
                            # Pre-initialize row to prevent "possibly unbound" Pylance warning
                            row: Dict[str, Any] = {}
                            
                            for txt_type, hyp_text in text_variants:
                                row = {
                                    "tool": tool_name, "model": model_name, "lang": lang or "N/A", 
                                    "device": device, "song": track_path.stem, "text_type": txt_type, "rtf": rtf
                                }
                                
                                for tier_num, clean_func in enumerate(tiers, 1):
                                    r_clean = clean_func(ref_raw)
                                    h_clean = clean_func(hyp_text)
                                    metrics = self.get_detailed_metrics(r_clean, h_clean)
                                    
                                    row[f"Tier{tier_num}_WER"] = metrics["wer"]
                                    row[f"Tier{tier_num}_CER"] = metrics["cer"]
                                    
                                    if tier_num == 4: # Store acoustic baseline errors
                                        row["substitutions"] = metrics["w_sub"]
                                        row["deletions"] = metrics["w_del"]
                                        row["insertions"] = metrics["w_ins"]

                                all_metrics = [m for m in all_metrics if not (m.get('tool')==tool_name and m.get('model')==model_name and m.get('song')==track_path.stem and m.get('lang')==(lang or "N/A") and m.get('text_type')==txt_type and m.get('device')==device)]
                                all_metrics.append(row)
                                pd.DataFrame(all_metrics).to_csv(self.csv_path, index=False)
                            
                            # Print summary and resolve "row" warning by ensuring row is not empty
                            if row:
                                print(f" -> {track_path.stem} processed. T4 Acoustic WER: {row.get('Tier4_WER', 1.0)*100:.1f}%")

                            # --- VISUAL ALIGNMENT TRIGGER ---
                            if self.args.align_track_idx is not None and track_idx == self.args.align_track_idx:
                                # We use Tier 2 for visualization as requested in the previous spec
                                vis_ref = self.tier_2_punctuation_strip(ref_raw)
                                vis_hyp = self.tier_2_punctuation_strip(hyp_time)
                                tool_info = f"{track_path.stem} | {tool_name} ({model_name}) | {device}"
                                self.generate_alignment_visualization(vis_ref, vis_hyp, tool_info)

                    if transcriber: del transcriber
                    gc.collect()
                    if device == "cuda": torch.cuda.empty_cache()

    def generate_summaries(self):
        if not self.csv_path.exists(): 
            print("[!] No CSV data found to summarize.")
            return
            
        df = pd.read_csv(self.csv_path)
        
        agg_dict = {'rtf': 'mean'}
        for i in range(1, 6):
            agg_dict[f'Tier{i}_WER'] = 'mean'
            agg_dict[f'Tier{i}_CER'] = 'mean'
            
        summary = df.groupby(['tool', 'model', 'text_type']).agg(agg_dict).reset_index()
        summary.to_json(self.json_path, orient='records', indent=4)
        
        # 3. Generate LaTeX Table
        with open(self.latex_path, 'w', encoding='utf-8') as f:
            f.write("\\begin{table}[hbt!]\n\\centering\n")
            f.write("\\caption{Mean WER and CER Across 5 Normalization Tiers}\n")
            f.write("\\resizebox{\\textwidth}{!}{\n")
            f.write("\\begin{tabular}{ll|c|ccccc|ccccc}\n\\toprule\n")
            f.write(" & & & \\multicolumn{5}{c|}{\\textbf{WER (\\%)}} & \\multicolumn{5}{c}{\\textbf{CER (\\%)}} \\\\\n")
            f.write("\\textbf{Tool} & \\textbf{Model} & \\textbf{Type} & T1 & T2 & T3 & T4 & T5 & T1 & T2 & T3 & T4 & T5 \\\\\n\\midrule\n")
            
            for _, row_data in summary.iterrows():
                wer_vals = [f"{row_data[f'Tier{i}_WER']*100:.2f}" for i in range(1, 6)]
                cer_vals = [f"{row_data[f'Tier{i}_CER']*100:.2f}" for i in range(1, 6)]
                type_short = "Uni" if row_data['text_type'] == "unified" else ("Cont" if row_data['text_type'] == "continuous" else "Time")
                
                f.write(f"{row_data['tool']} & {row_data['model']} & {type_short} & " + " & ".join(wer_vals) + " & " + " & ".join(cer_vals) + " \\\\\n")
                
            f.write("\\bottomrule\n\\end{tabular}\n}\n\\end{table}\n")
            
        print(f"\n[+] Saved LaTeX table to {self.latex_path}")

        # 4. Generate Graphs
        if PLOTTING_AVAILABLE and plt is not None and sns is not None:
            sns.set_theme(style="whitegrid")
            
            summary['System'] = summary['tool'] + "\n(" + summary['model'] + ")\n" + summary['text_type']
            
            # --- RTF Plot ---
            plt.figure(figsize=(10, 6))
            ax = sns.barplot(data=summary, x='System', y='rtf', palette="viridis")
            
            plt.title("Mean Real-Time Factor (RTF) per Model", fontsize=14, pad=15, weight='bold')
            plt.ylabel("RTF (Lower is faster)", fontsize=12, labelpad=10)
            plt.xlabel("Evaluated Models", fontsize=12, labelpad=10)
            plt.xticks(rotation=45, ha="right")
            
            # Headroom and Labels
            ax.set_ylim(0, summary['rtf'].max() * 1.15)
            for container in getattr(ax, 'containers', []):
                ax.bar_label(
                    container, fmt='%.2f', padding=3, weight='bold', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.9)
                )

            plt.tight_layout()
            plt.savefig(self.plots_dir / "Trans_RTF_Comparison.png", dpi=300)
            plt.close()
            
            # =====================================================================
            # SINGLE TIER SELECTION & Y-AXIS SYNC FOR WER/CER
            # =====================================================================
            # For French Rap, Tier 4 (Acoustic Robust) is ideal. 
            # It strips accents and splits apostrophes/hyphens, preventing models 
            # from being penalized for orthographic differences of the same sounds.
            target_tier = 4
            
            # Extract the specific tier data and convert to percentages
            summary['WER_Plot'] = summary[f'Tier{target_tier}_WER'] * 100
            summary['CER_Plot'] = summary[f'Tier{target_tier}_CER'] * 100
            
            # Calculate global maximum across BOTH WER and CER to sync the y-axis
            global_max_err = max(summary['WER_Plot'].max(), summary['CER_Plot'].max())
            shared_y_limit = global_max_err * 1.15
            
            # --- WER Plot ---
            plt.figure(figsize=(10, 6))
            ax_wer = sns.barplot(data=summary, x='System', y='WER_Plot', palette="rocket")
            
            plt.title(f"Mean Word Error Rate (WER) - Tier {target_tier} (Acoustic Robust)", fontsize=14, pad=15, weight='bold')
            plt.ylabel("WER (%)", fontsize=12, labelpad=10)
            plt.xlabel("Evaluated Models", fontsize=12, labelpad=10)
            plt.xticks(rotation=45, ha="right")
            
            # Apply the shared Y-axis limit
            ax_wer.set_ylim(0, shared_y_limit)
            for container in getattr(ax_wer, 'containers', []):
                ax_wer.bar_label(
                    container, fmt='%.1f', padding=3, weight='bold', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.9)
                )

            plt.tight_layout()
            plt.savefig(self.plots_dir / f"WER_Tier{target_tier}_Comparison.png", dpi=300)
            plt.close()
            
            # --- CER Plot ---
            plt.figure(figsize=(10, 6))
            ax_cer = sns.barplot(data=summary, x='System', y='CER_Plot', palette="mako")
            
            plt.title(f"Mean Character Error Rate (CER) - Tier {target_tier} (Acoustic Robust)", fontsize=14, pad=15, weight='bold')
            plt.ylabel("CER (%)", fontsize=12, labelpad=10)
            plt.xlabel("Evaluated Models", fontsize=12, labelpad=10)
            plt.xticks(rotation=45, ha="right")
            
            # Apply the shared Y-axis limit
            ax_cer.set_ylim(0, shared_y_limit)
            for container in getattr(ax_cer, 'containers', []):
                ax_cer.bar_label(
                    container, fmt='%.1f', padding=3, weight='bold', fontsize=11,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.9)
                )

            plt.tight_layout()
            plt.savefig(self.plots_dir / f"CER_Tier{target_tier}_Comparison.png", dpi=300)
            plt.close()
            
            print(f"[+] Saved RTF, WER, and CER graphs (synced y-axis) to {self.plots_dir}/")
            
        print("\n--- EVALUATION COMPLETE ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="Evaluation/dataset_muni")
    parser.add_argument("--separated_path", type=str, default="Evaluation/separated_muni")
    parser.add_argument("--device", type=str, default="Auto")
    parser.add_argument("--num_tracks", type=int, default=None)
    parser.add_argument("--align_track_idx", type=int, default=None, help="Index of the track (0-based) to print visual alignment for.")
    parser.add_argument("--clean_run", action="store_true", help="Deletes transcription_results and runs a fresh evaluation.")
    parser.add_argument("--force_reseparate", action="store_true", help="Deletes the separated audio folder to force re-separation.")
    parser.add_argument("--model_path", type=str, default="./Models")
    parser.add_argument("--run_separation", nargs="?", const="demucs-htdemucs", default=None)
    parser.add_argument("--tools", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--whisper_langs", nargs="+", default=["auto", "fr"], help="Languages to force Whisper to use.")
    parser.add_argument("--skip_transcription", action="store_true")
    parser.add_argument("--fetch_lyrics", action="store_true")
    parser.add_argument("--genius_token", type=str, default="")
    parser.add_argument("--debug", action="store_true", help="Print granular execution times and error diagnostics.")
    
    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

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