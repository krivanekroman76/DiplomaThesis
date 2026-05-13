"""
================================================================================
AUDIO SEPARATION EVALUATION PIPELINE
================================================================================
AUTHOR: Bc. Roman Křivánek

USAGE GUIDE:
1.  STANDARD EVAL:  python evaluate.py --musdb_path "path/to/musdb"
2.  CLEAN START:    python evaluate.py --clean_run (Deletes previous results)
3.  REGEN GRAPHS:   python evaluate.py --only_reports (Uses existing CSV)
4.  HW TARGETING:   python evaluate.py --device cuda (Forces GPU only)
5.  TEMP BENCHMARK: python evaluate.py --flag_download (Measures DL time)

LOGIC:
- Measures 'Waking Time' (Cold Start) vs 'Pure Separation' (Warm RTF).
- Spleeter is restricted to CPU on Windows unless TensorFlow-GPU is verified.
- Memory is forcefully cleared between every Model/Hardware switch.
================================================================================
"""

import os
import sys
import time
import json
import argparse
import platform
import tempfile
from pathlib import Path
import cpuinfo
from contextlib import ExitStack
import shutil
import logging
import gc
from typing import Any, Dict, List
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import soundfile as sf
import museval
import matplotlib.pyplot as plt
import seaborn as sns
import musdb 

# Setup Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR) # Suppress TF log spam

sys.path.append(str(Path(__file__).parent.parent))
from separators.utils import setup_ffmpeg_environment, download_required_models
SpleeterSeparator = None
DemucsSeparator = None
OpenUnmixSeparator = None
try:
    from separators.spleeter_separator import SpleeterSeparator
    from separators.demucs_separator import DemucsSeparator
    from separators.openunmix_separator import OpenUnmixSeparator
except ImportError as e:
    print(f"Error: Could not find separator modules: {e}")

class SeparationEvaluator:
    def __init__(self, args):
        self.args = args
        self.wav_dir = Path("Evaluation/evaluated_musdb")
        self.results_dir = Path("Evaluation/evaluated_results")
        self.csv_path = self.results_dir / "raw_metrics_per_song.csv"
        self.json_path = self.results_dir / "summary_results.json"
        self.model_dir = Path(args.model_path).absolute()
        
        for p in [self.wav_dir, self.results_dir, self.model_dir]: p.mkdir(parents=True, exist_ok=True)
        
        setup_ffmpeg_environment()
        self.devices_to_test = self._resolve_devices(args.device)
        self.gpu_available_tf = len(tf.config.list_physical_devices('GPU')) > 0

    def _resolve_devices(self, device_arg):
        device_arg = device_arg.lower()
        if device_arg == "auto":
            return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"] 
        return ["cuda"] if device_arg in ["gpu", "cuda"] and torch.cuda.is_available() else ["cpu"]

    def _get_cpu_name(self):
        try: return cpuinfo.get_cpu_info().get('brand_raw', 'Unknown CPU')
        except: return platform.processor()

    def _clear_memory(self):
        """Forcefully clears RAM and VRAM."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        tf.keras.backend.clear_session()

    def run_evaluation_pipeline(self):
        if self.args.clean_run:
            for p in [self.results_dir, self.wav_dir]: 
                if p.exists(): shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
            
        df_existing = pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame()
        # Tell Pylance to treat the incoming data as a generic list first
        all_metrics: Any = df_existing.to_dict('records') if not df_existing.empty else []

        mus = musdb.DB(root=self.args.musdb_path, subsets="test", is_wav=True)
        test_samples = mus.tracks[:self.args.num_tracks] if self.args.num_tracks else mus.tracks
        tools = [self.args.tool] if self.args.tool else ["Demucs", "OpenUnmix", "Spleeter"]

        for tool_name in tools:
            models = self._get_models_for_tool(tool_name)
            for model_name in models:
                with ExitStack() as stack:
                    active_model_dir = stack.enter_context(tempfile.TemporaryDirectory()) if self.args.flag_download else str(self.model_dir)
                    os.environ["TORCH_HOME"] = active_model_dir
                    
                    start_dl = time.time()
                    download_required_models(active_model_dir, tool_name, [model_name] if tool_name == "Demucs" else None)
                    dl_time = (time.time() - start_dl) if self.args.flag_download else 0

                    for current_device in self.devices_to_test:
                        # Logic: Skip Spleeter-CUDA if TF can't see the GPU
                        if tool_name == "Spleeter" and current_device == "cuda" and not self.gpu_available_tf:
                            print(f"[!] Skipping Spleeter-CUDA: TensorFlow GPU support not detected.")
                            continue

                        device_name = torch.cuda.get_device_name(0) if current_device == "cuda" else self._get_cpu_name()
                        
                        print(f"\n--- Benchmark: {tool_name} [{model_name}] on {current_device.upper()} ---")
                        start_init = time.time()
                        # 1. Define the map (using actual classes)
                        separator_map = {
                            "Spleeter": SpleeterSeparator, 
                            "Demucs": DemucsSeparator,       
                            "OpenUnmix": OpenUnmixSeparator 
                        }

                        # 2. Get the class safely
                        sep_class = separator_map.get(tool_name)

                        # 3. Check if it exists before calling ()
                        if sep_class is None:
                            print(f"[!] Error: {tool_name} is not a valid separator. Skipping...")
                            continue

                        # 4. Initialize the instance
                        separator = sep_class()
                        init_overhead = time.time() - start_init 
                        
                        run_output_dir = self.wav_dir / f"{tool_name}_{model_name}_{current_device}"
                        run_output_dir.mkdir(parents=True, exist_ok=True)
                        is_first_track = True

                        for track in test_samples:
                            if self._is_already_evaluated(df_existing, tool_name, model_name, current_device, track.name):
                                continue

                            timing = {"waking": 0.0, "sep": 0.0}
                            callback_hit = [False]
                            start_ts = [0.0]

                            def progress_cb(**kwargs):
                                if not callback_hit[0]:
                                    start_ts[0], callback_hit[0] = time.time(), True

                            start_call = time.time()
                            try:
                                result = self._execute_separation(separator, tool_name, model_name, track, run_output_dir, current_device, cb=progress_cb)
                                end_call = time.time()

                                timing["waking"] = (start_ts[0] - start_call) if callback_hit[0] else 0.0
                                timing["sep"] = (end_call - start_ts[0]) if callback_hit[0] else (end_call - start_call)
                                
                                # Cold Start is assigned to the first track only
                                waking_cost = timing["waking"] if is_first_track else 0.0
                                rtf = timing["sep"] / track.duration

                                if result and result[0]:
                                    metrics = self._calculate_museval(track, run_output_dir / result[1], run_output_dir / result[2])
                                    row = {
                                        "tool": tool_name, "model": model_name, "device_type": current_device.upper(),
                                        "device_name": device_name, "song": track.name, "duration": track.duration,
                                        "init_overhead_sec": init_overhead, "waking_time_sec": waking_cost,
                                        "pure_sep_time_sec": timing["sep"], "rtf": rtf, **metrics
                                    }
                                    # Convert just the single new row to a DF
                                    new_row_df = pd.DataFrame([row])

                                    # Append to CSV, only write header if the file doesn't exist yet
                                    new_row_df.to_csv(self.csv_path, mode='a', index=False, header=not self.csv_path.exists())
                                    is_first_track = False
                                    print(f"  > {track.name}: RTF={rtf:.3f}, SDR_V={metrics['SDR_vocals']:.2f}")

                            except Exception as e:
                                print(f"Error on {track.name}: {e}")

                        self._append_to_json_summary(pd.DataFrame(all_metrics), tool_name, model_name, current_device.upper())
                        self._clear_memory() # Clean up after the device loop

        self.generate_reports(all_metrics)

    def _execute_separation(self, sep, tool, model, track, out_dir, device, cb):
        if tool == "Spleeter":
            return sep.separate(track.path, track.name, str(out_dir), str(out_dir), "2stems", "wav", 44100, "128k", device, progress_callback=cb)
        elif tool == "Demucs":
            return sep.separate(track.path, track.name, str(out_dir), str(out_dir), model, 2, "wav", 44100, "128k", 16, 2, 1, 0.1, device, progress_callback=cb)
        else: # OpenUnmix
            return sep.separate(track.path, track.name, str(out_dir), str(out_dir), model, 2, "wav", 44100, "128k", device, progress_callback=cb)

    def _calculate_museval(self, track, v_path, i_path):
        est_v, _ = sf.read(v_path); est_i, _ = sf.read(i_path)
        ref_v, ref_i = track.targets['vocals'].audio, track.targets['accompaniment'].audio
        L = min(len(est_v), len(ref_v), len(est_i), len(ref_i))
        sdr, isr, sir, sar = museval.evaluate([ref_v[:L], ref_i[:L]], [est_v[:L], est_i[:L]])
        return {"SDR_vocals": np.nanmedian(sdr[0]), "SDR_instr": np.nanmedian(sdr[1])}

    def _is_already_evaluated(self, df, tool, model, dev, song):
        if df.empty: return False
        return not df[(df['tool']==tool)&(df['model']==model)&(df['device_type']==dev.upper())&(df['song']==song)].empty

    def _get_models_for_tool(self, tool):
        # Fallback to standard models if settings.json is missing
        return {"Demucs": ["htdemucs"], "OpenUnmix": ["umxhq"], "Spleeter": ["2stems"]}.get(tool, ["2stems"])

    def _append_to_json_summary(self, df, tool, model, device):
        subset = df[(df['tool'] == tool) & (df['model'] == model) & (df['device_type'] == device)]
        if subset.empty: return
        means = subset.select_dtypes(include=[np.number]).mean().to_dict()
        data = json.load(open(self.json_path, 'r')) if self.json_path.exists() else {}
        data.setdefault(tool, {}).setdefault(model, {})[device] = means
        with open(self.json_path, 'w') as f: json.dump(data, f, indent=4)

    def generate_reports(self, data, from_csv=False):
        df = pd.read_csv(self.csv_path) if from_csv else pd.DataFrame(data)
        if df.empty: return
        summary = df.groupby(['tool', 'model', 'device_type']).mean(numeric_only=True)
        with open(self.results_dir / "latex_table.txt", "w") as f: f.write(summary.to_latex())
        
        graph_dir = self.results_dir / "graphs"; graph_dir.mkdir(exist_ok=True)
        self._plot_metric(df, 'rtf', "RTF (Lower is Better)", graph_dir / "RTF_Comparison.png", True)
        self._plot_metric(df, 'SDR_vocals', "SDR Vocals (Higher is Better)", graph_dir / "SDR_Comparison.png", False)

    def _plot_metric(self, df, col, title, path, horizontal_line):
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        df['Label'] = df['tool'] + "\n(" + df['model'] + ")"
        sns.barplot(data=df, x='Label', y=col, hue='device_type', palette="muted")
        if horizontal_line: plt.axhline(1.0, color='red', linestyle='--')
        plt.title(title); plt.tight_layout(); plt.savefig(path, dpi=300); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb_path", type=str, default="Evaluation/musdb18_test_samples")
    parser.add_argument("--model_path", type=str, default="./Models")
    parser.add_argument("--device", type=str, default="Auto")
    parser.add_argument("--num_tracks", type=int, default=None)
    parser.add_argument("--tool", type=str, default=None)
    parser.add_argument("--flag_download", action="store_true")
    parser.add_argument("--only_reports", action="store_true")
    parser.add_argument("--clean_run", action="store_true")
    args = parser.parse_args()
    evaluator = SeparationEvaluator(args)
    if args.only_reports: evaluator.generate_reports(None, from_csv=True)
    else: evaluator.run_evaluation_pipeline()