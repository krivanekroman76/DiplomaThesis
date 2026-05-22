"""
================================================================================
AUDIO SEPARATION EVALUATION PIPELINE
================================================================================
AUTHOR: Bc. Roman Křivánek

PREREQUISITES:
- This script scans 'settings.json' to determine which separation models to evaluate.
- Before running this script, run the main GUI app ('separation_app.py') at least
  once to auto-generate the 'settings.json' configuration file.
- You can change the enabled models or their execution order either directly through 
  the GUI or by manually editing the 'settings.json' file in your IDE.

USAGE GUIDE (Commands can be chained together!):
1.  STANDARD EVAL:  python Evaluation/evaluate_separation.py 
2.  CLEAN START:    python Evaluation/evaluate_separation.py --clean_run (Deletes previous results)
3.  REGEN GRAPHS:   python Evaluation/evaluate_separation.py --only_reports (Uses existing CSV)
4.  HW TARGETING:   python Evaluation/evaluate_separation.py --device cuda (Forces GPU only)
5.  TEMP BENCHMARK: python Evaluation/evaluate_separation.py --flag_download (Measures DL time)
6.  LIMIT TRACKS:   python Evaluation/evaluate_separation.py --num_tracks 2 (Evaluates only the first N tracks)
7.  TARGET TOOL:    python Evaluation/evaluate_separation.py --tool Demucs (Evaluates only a specific tool)
8.  MODEL PATH:     python Evaluation/evaluate_separation.py --model_path "./Models" (Custom weights folder)

EXAMPLE OF CHAINED COMMAND:
python Evaluation/evaluate_separation.py --clean_run --device cuda --num_tracks 5 --tool OpenUnmix --debug
Arguments:
--musdb_path "path/to/musdb"
--clean_run
--device cuda (cpu)
--flag_download
--num_tracks 2
--tool Demucs
--model_path

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
from typing import Any, Dict, List, Tuple
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
import soundfile as sf
import museval
import matplotlib.pyplot as plt
import seaborn as sns
import musdb 

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
        """
        Initializes the Evaluation Pipeline.
        
        Args:
            args (argparse.Namespace): The parsed command-line arguments.
            
        Internal Actions:
            - Sets up file paths for input, output, and caching.
            - Configures the logging module based on the --debug flag.
            - Injects FFmpeg paths into the system environment.
            - Detects available hardware (CPU/GPU) for PyTorch and TensorFlow.
        """
        self.args = args
        self.wav_dir = Path("Evaluation/evaluated_musdb")
        self.results_dir = Path("Evaluation/evaluated_results")
        self.csv_path = self.results_dir / "raw_metrics_per_song.csv"
        self.json_path = self.results_dir / "summary_results.json"
        self.model_dir = Path(args.model_path).absolute()
        
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        
        self._ensure_directories()
        setup_ffmpeg_environment()
        self.devices_to_test = self._resolve_devices(args.device)
        self.gpu_available_tf = len(tf.config.list_physical_devices('GPU')) > 0
        
        logging.debug(f"Evaluator initialized. Target Devices: {self.devices_to_test}")

    def _ensure_directories(self):
        """
        Verifies that all necessary working directories exist.
        
        Internal Actions:
            - Loops through the WAV output, results, and model directories.
            - Creates them (and any parent directories) if they are missing.
        """
        for p in [self.wav_dir, self.results_dir, self.model_dir]: 
            p.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Ensured directory exists: {p}")

    def _resolve_devices(self, device_arg: str) -> List[str]:
        """
        Determines which hardware devices to use for the benchmark.
        
        Args:
            device_arg (str): The device requested by the user ('auto', 'cuda', 'cpu').
            
        Returns:
            List[str]: A list containing the valid devices to iterate over 
                       (e.g., ['cuda', 'cpu'] or just ['cpu']).
        """
        device_arg = device_arg.lower()
        if device_arg == "auto":
            return ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"] 
        return ["cuda"] if device_arg in ["gpu", "cuda"] and torch.cuda.is_available() else ["cpu"]

    def _get_cpu_name(self) -> str:
        """
        Attempts to fetch the human-readable brand name of the host CPU.
        
        Returns:
            str: The exact CPU model (e.g., 'AMD Ryzen 9 5900X') or a generic fallback.
        """
        try: return cpuinfo.get_cpu_info().get('brand_raw', 'Unknown CPU')
        except: return platform.processor()

    def _clear_memory(self):
        """
        Aggressively flushes system RAM and GPU VRAM.
        
        Internal Actions:
            - Calls Python's garbage collector.
            - Empties PyTorch's CUDA cache and synchronizes the GPU threads.
            - Clears TensorFlow's Keras backend session.
        """
        logging.debug("Clearing RAM and VRAM...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        tf.keras.backend.clear_session()

    def run_evaluation_pipeline(self):
        """
        The main execution loop for evaluating the models.
        
        Internal Actions:
            - Parses the MUSDB18 dataset.
            - Downloads models if necessary (or simulates a clean download via a temp folder).
            - Instantiates separation classes (Demucs, Spleeter, OpenUnmix).
            - Processes audio tracks, capturing execution times (Cold Start vs Pure RTF).
            - Calculates Museval metrics (SDR) and writes row-by-row to a CSV.
            - Triggers report and graph generation upon completion.
        """
        self._ensure_directories()

        if self.args.clean_run:
            logging.info("Clean run flag detected. Deleting previous results...")
            for p in [self.results_dir, self.wav_dir]: 
                if p.exists(): shutil.rmtree(p)
                p.mkdir(parents=True, exist_ok=True)
            
        df_existing = pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame()
        all_metrics: Any = df_existing.to_dict('records') if not df_existing.empty else []

        logging.info(f"Loading MUSDB18 dataset from: {self.args.musdb_path}")
        mus = musdb.DB(root=self.args.musdb_path, subsets="test", is_wav=True)
        test_samples = mus.tracks[:self.args.num_tracks] if self.args.num_tracks else mus.tracks
        
        # --- NEW CLI OVERRIDE LOGIC ---
        if self.args.run_separation:
            try:
                # We split on the FIRST hyphen only, just in case a model name contains hyphens (e.g., 'my-custom-model')
                target_tool, target_model = self.args.run_separation.split('-', 1)
                
                # Case-insensitive tool matching for better CLI UX
                tool_map = {"demucs": "Demucs", "spleeter": "Spleeter", "openunmix": "OpenUnmix"}
                target_tool = tool_map.get(target_tool.lower(), target_tool)
                
                tools_and_models = [(target_tool, [target_model])]
                logging.info(f"CLI Override active: Forcing evaluation of {target_tool} -> {target_model}")
            except ValueError:
                logging.error("[!] Invalid format for --run_separation. Expected Tool-Model (e.g., Demucs-htdemucs_ft). Exiting.")
                return
        else:
            # Standard settings.json / fallback behavior
            tools_to_check = [self.args.tool] if self.args.tool else ["Demucs", "OpenUnmix", "Spleeter"]
            
            # Case-insensitive mapping to match the JSON keys and internal logic
            tool_map = {"demucs": "Demucs", "spleeter": "Spleeter", "openunmix": "OpenUnmix"}
            tools_to_check = [tool_map.get(t.lower(), t) for t in tools_to_check]
            
            tools_and_models = [(t, self._get_models_for_tool(t)) for t in tools_to_check]
        
        # Process the resolved targets
        for tool_name, models in tools_and_models:
            # Type guard: Ensure tool_name is a string for the type checker
            if not isinstance(tool_name, str):
                continue
                
            for model_name in models:
                # Type guard: Ensure model_name is a string for the type checker
                if not isinstance(model_name, str):
                    continue
                    
                with ExitStack() as stack:
                    active_model_dir = stack.enter_context(tempfile.TemporaryDirectory()) if self.args.flag_download else str(self.model_dir)
                    os.environ["TORCH_HOME"] = active_model_dir
                    
                    start_dl = time.time()
                    download_required_models(active_model_dir, tool_name, [model_name] if tool_name == "Demucs" else None)
                    dl_time = (time.time() - start_dl) if self.args.flag_download else 0

                    for current_device in self.devices_to_test:
                        if tool_name == "Spleeter" and current_device == "cuda" and not self.gpu_available_tf:
                            logging.warning("[!] Skipping Spleeter-CUDA: TensorFlow GPU support not detected.")
                            continue

                        # Determine display names for logging and CSV
                        display_device = "GPU CUDA" if current_device == "cuda" else "CPU"
                        device_name = torch.cuda.get_device_name(0) if current_device == "cuda" else self._get_cpu_name()
                        
                        logging.info(f"\n--- Benchmark: {tool_name} [{model_name}] on {display_device} ---")
                        logging.debug(f"Hardware Name: {device_name}")
                        start_init = time.time()
                        
                        separator_map = {
                            "Spleeter": SpleeterSeparator, 
                            "Demucs": DemucsSeparator,       
                            "OpenUnmix": OpenUnmixSeparator 
                        }

                        sep_class = separator_map.get(tool_name)
                        if sep_class is None:
                            logging.error(f"[!] Error: {tool_name} is not a valid separator. Skipping...")
                            continue

                        separator = sep_class()
                        init_overhead = time.time() - start_init 
                        logging.debug(f"Tool Initialization Overhead: {init_overhead:.3f} seconds")
                        
                        run_output_dir = self.wav_dir / f"{tool_name}_{model_name}_{current_device}"
                        run_output_dir.mkdir(parents=True, exist_ok=True)
                        is_first_track = True

                        for track in test_samples:
                            if self._is_already_evaluated(df_existing, tool_name, model_name, display_device, track.name):
                                logging.debug(f"Skipping {track.name} (Already Evaluated)")
                                continue

                            timing = {"waking": 0.0, "sep": 0.0}
                            callback_hit = [False]
                            start_ts = [0.0]

                            def progress_cb(*args, **kwargs):
                                if not callback_hit[0]:
                                    start_ts[0], callback_hit[0] = time.time(), True

                            logging.debug(f"Processing track: {track.name} | Duration: {track.duration:.2f}s | Original SR: {track.rate}")
                            start_call = time.time()
                            try:
                                result = self._execute_separation(separator, tool_name, model_name, track, run_output_dir, current_device, cb=progress_cb)
                                end_call = time.time()

                                timing["waking"] = (start_ts[0] - start_call) if callback_hit[0] else 0.0
                                timing["sep"] = (end_call - start_ts[0]) if callback_hit[0] else (end_call - start_call)
                                
                                waking_cost = timing["waking"] if is_first_track else 0.0
                                rtf = timing["sep"] / track.duration

                                if self.args.debug:
                                    logging.debug(f"  [Timing] Duration: {track.duration:.2f}s | Waking: {waking_cost:.3f}s | Pure Sep: {timing['sep']:.3f}s | RTF: {rtf:.3f}")

                                if result and result[0]:
                                    v_path, i_path = run_output_dir / result[1], run_output_dir / result[2]
                                    
                                    if self.args.debug:
                                        try:
                                            v_info, i_info = sf.info(v_path), sf.info(i_path)
                                            v_size, i_size = v_path.stat().st_size / (1024*1024), i_path.stat().st_size / (1024*1024)
                                            logging.debug(f"  [Output Data] Vocals: {v_info.samplerate}Hz, {v_info.subtype}, {v_size:.2f}MB")
                                            logging.debug(f"  [Output Data] Instr: {i_info.samplerate}Hz, {i_info.subtype}, {i_size:.2f}MB")
                                        except Exception as e:
                                            logging.debug(f"  [Output Data] Could not read file info for debugging: {e}")

                                    metrics = self._calculate_museval(track, v_path, i_path)
                                    row = {
                                        "tool": tool_name, "model": model_name, "device_type": display_device,
                                        "device_name": device_name, "song": track.name, "duration": track.duration,
                                        "init_overhead_sec": init_overhead, "waking_time_sec": waking_cost,
                                        "pure_sep_time_sec": timing["sep"], "rtf": rtf, **metrics
                                    }
                                    all_metrics.append(row) 
                                    
                                    new_row_df = pd.DataFrame([row])
                                    new_row_df.to_csv(self.csv_path, mode='a', index=False, header=not self.csv_path.exists())
                                    is_first_track = False
                                    logging.info(f"  > {track.name}: RTF={rtf:.3f}, SDR_V={metrics['SDR_vocals']:.2f}")

                            except Exception as e:
                                logging.error(f"Error on {track.name}: {e}")

                        self._append_to_json_summary(pd.DataFrame(all_metrics), tool_name, model_name, display_device)
                        self._clear_memory()

        self.generate_reports(all_metrics)

    def _execute_separation(self, sep, tool: str, model: str, track: musdb.audio_classes.MultiTrack, out_dir: Path, device: str, cb) -> Tuple[bool, str, str]:
        """
        Delegates the separation task to the specific model implementation.
        
        Args:
            sep: The initialized separator object (e.g., DemucsSeparator instance).
            tool (str): Name of the toolkit ("Spleeter", "Demucs", "OpenUnmix").
            model (str): Specific model variant (e.g., "htdemucs").
            track (musdb.audio_classes.MultiTrack): The MUSDB track object to separate.
            out_dir (Path): Where to save the output files.
            device (str): "cuda" or "cpu".
            cb (Callable): The callback function to track real-time processing start.
            
        Returns:
            Tuple[bool, str, str]: A tuple containing a success flag, the vocals filename, 
                                   and the instrumentals filename.
        """
        if tool == "Spleeter":
            return sep.separate(input_path=track.path, song_name=track.name, vocals_folder=str(out_dir), instr_folder=str(out_dir), 
                                channels="Stereo", fmt="wav", sr=44100, bitrate="128k",bit_depth="32-bit", device_choice=device, progress_callback=cb)
        elif tool == "Demucs":
            return sep.separate(input_path=track.path, song_name=track.name, vocals_folder=str(out_dir), instr_folder=str(out_dir), 
                                model=model, channels="Stereo", fmt="wav", sr=44100, bitrate="128k",  bit_depth="32-bit",
                                flac_compression=5, shifts=1, overlap=0.1, device_choice=device, progress_callback=cb)
        else: # OpenUnmix
            return sep.separate(input_path=track.path, song_name=track.name, vocals_folder=str(out_dir), instr_folder=str(out_dir), 
                                model=model, channels="Stereo", fmt="wav", sr=44100, bitrate="128k", bit_depth="32-bit", device_choice=device, progress_callback=cb)
                 
    def _calculate_museval(self, track: musdb.audio_classes.MultiTrack, v_path: Path, i_path: Path) -> Dict[str, float]:
        """
        Computes SDR, SIR, SAR, and ISR using the bss_eval metric.
        
        Args:
            track (musdb.audio_classes.MultiTrack): The MUSDB reference track containing ground truth.
            v_path (Path): Path to the estimated vocals file.
            i_path (Path): Path to the estimated instrumentals (accompaniment) file.
            
        Returns:
            Dict[str, float]: Dictionary containing median metric values for vocals and instrumentals.
        """
        est_v, _ = sf.read(v_path)
        est_i, _ = sf.read(i_path)
        
        # Guard clause to satisfy Pylance type checking
        if track.targets is None:
            raise ValueError(f"Ground truth targets missing for track: {track.name}")
            
        ref_v = track.targets['vocals'].audio
        ref_i = track.targets['accompaniment'].audio
        
        # Ensure lengths match
        L = min(len(est_v), len(ref_v), len(est_i), len(ref_i))
        
        # museval returns arrays of shape (n_targets, n_frames)
        # Index 0 is targets (vocals), Index 1 is accompaniment (instrumentals)
        sdr, isr, sir, sar = museval.evaluate([ref_v[:L], ref_i[:L]], [est_v[:L], est_i[:L]])
        
        return {
            "SDR_vocals": np.nanmedian(sdr[0]),
            "SDR_instr": np.nanmedian(sdr[1]),
            "SIR_vocals": np.nanmedian(sir[0]),
            "SIR_instr": np.nanmedian(sir[1]),
            "SAR_vocals": np.nanmedian(sar[0]),
            "SAR_instr": np.nanmedian(sar[1]),
            "ISR_vocals": np.nanmedian(isr[0]),
            "ISR_instr": np.nanmedian(isr[1])
        }

    def _is_already_evaluated(self, df: pd.DataFrame, tool: str, model: str, dev: str, song: str) -> bool:
        """
        Checks if a specific combination of tool, model, hardware, and song 
        has already been processed in the existing CSV file.
        
        Args:
            df (pd.DataFrame): The dataframe containing past evaluations.
            tool (str): The separation tool.
            model (str): The model variant.
            dev (str): Hardware device used (e.g., 'GPU CUDA' or 'CPU').
            song (str): Name of the song.
            
        Returns:
            bool: True if the song was already processed, False otherwise.
        """
        if df.empty or 'tool' not in df.columns: 
            return False
        return not df[(df['tool']==tool)&(df['model']==model)&(df['device_type']==dev)&(df['song']==song)].empty

    def _get_models_for_tool(self, tool: str) -> List[str]:
        """
        Retrieves the enabled models associated with a tool from settings.json.
        Falls back to defaults if settings.json is missing or structurally invalid.
        """
        # Hardcoded fallbacks in case settings.json isn't available or keys are empty
        fallback_defaults = {
            "Demucs": ["htdemucs"], 
            "OpenUnmix": ["umxhq"], 
            "Spleeter": ["2stems"]
        }
        
        settings_path = Path("settings.json")
        
        if settings_path.exists():
            try:
                with open(settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                
                # 1. Extract the separator models
                separator_models = settings.get("separator_models", {})

                # 2. Check and default Spleeter
                # If Spleeter is missing or the list is empty ([]), default it to ["2stems"]
                if not separator_models.get("Spleeter"):
                    logging.info("Models list for Spleeter is empty in settings.json. Defaulting to ['2stems'].")
                    separator_models["Spleeter"] = ["2stems"]
                
                # 3. Check for the requested tool inside the updated dictionary
                if tool in separator_models:
                    models_list = separator_models[tool]
                    
                    # Handle Case A: Direct list -> {"separator_models": {"Demucs": [...]}}
                    if isinstance(models_list, list):
                        if models_list:  # Ensure the list isn't empty
                            logging.debug(f"Loaded models for {tool} from settings.json: {models_list}")
                            return models_list
                        else:
                            logging.info(f"Models list for {tool} is empty in settings.json. Using fallback default.")
                    
                    # Handle Case B: Nested dictionary structures
                    elif isinstance(models_list, dict):
                        for key in ['enabled_models', 'models', 'selected_models']:
                            if key in models_list and isinstance(models_list[key], list) and models_list[key]:
                                logging.debug(f"Loaded nested models for {tool} from settings.json: {models_list[key]}")
                                return models_list[key]
                                
            except Exception as e:
                logging.warning(f"Failed to parse settings.json ({e}). Falling back to default configuration.")
        else:
            logging.warning("settings.json not found! Using default evaluation models.")

        # 4. Final Fallback if the tool wasn't found or was invalid
        return fallback_defaults.get(tool, ["2stems"])

    def _append_to_json_summary(self, df: pd.DataFrame, tool: str, model: str, device: str):
        """
        Calculates the mean averages for a specific run configuration and 
        appends the data to a JSON dictionary.
        
        Args:
            df (pd.DataFrame): Dataframe containing all row-level metrics.
            tool (str): The tool being aggregated.
            model (str): The model being aggregated.
            device (str): The hardware device being aggregated.
            
        Internal Actions:
            - Reads the existing JSON summary file.
            - Updates the nested dictionary structure (Tool -> Model -> Device -> Metrics).
            - Writes the structure back to disk.
        """
        if df.empty or 'tool' not in df.columns: return
            
        subset = df[(df['tool'] == tool) & (df['model'] == model) & (df['device_type'] == device)]
        if subset.empty: return
            
        means = subset.select_dtypes(include=[np.number]).mean().to_dict()
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        if self.json_path.exists():
            try:
                with open(self.json_path, 'r') as f: data = json.load(f)
            except json.JSONDecodeError:
                data = {}
                
        data.setdefault(tool, {}).setdefault(model, {})[device] = means
        with open(self.json_path, 'w') as f: json.dump(data, f, indent=4)

    def generate_reports(self, data: Any, from_csv: bool = False, clean_run: bool = False):
        """
        Transforms raw evaluation metrics into visual graphs and academic tables.
        
        Args:
            data (Any): In-memory list of dictionaries/dataframe of the metrics.
            from_csv (bool): If true, bypasses the in-memory list and reads directly from the CSV.
            clean_run (bool): If true, deletes old tables, JSONs, and graphs before regenerating.
        """
        if clean_run:
            logging.info("Clean run detected. Removing old report files...")
            # Delete flat files
            for filename in ["latex_table.txt", "summary_results.json"]:
                file_path = self.results_dir / filename
                if file_path.exists():
                    file_path.unlink()
            
            # Delete entire graphs directory and its contents
            graph_dir = self.results_dir / "graphs"
            if graph_dir.exists():
                shutil.rmtree(graph_dir)

        df = pd.read_csv(self.csv_path) if from_csv else pd.DataFrame(data)
        if df.empty or 'tool' not in df.columns: 
            logging.warning("No data available to generate reports.")
            return

        if self.args.debug:   
            logging.debug(f"\n🚨 EXACT COLUMNS AVAILABLE IN CSV: {df.columns.tolist()}\n")
            
        summary = df.groupby(['tool', 'model', 'device_type']).mean(numeric_only=True)
        
        # Write new LaTeX table
        with open(self.results_dir / "latex_table.txt", "w") as f: 
            f.write(summary.style.to_latex())
        
        # Write new JSON summary (assuming you want the grouped mean summary saved to JSON)
        summary.to_json(self.results_dir / "summary_results.json", orient="index")
        
        graph_dir = self.results_dir / "graphs"
        graph_dir.mkdir(exist_ok=True)
        
        # RTF depends on hardware, so we keep the hue (include_hue=True)
        self._plot_metric(df, 'rtf', "RTF (Lower is Better)", graph_dir / "RTF_Comparison.png", horizontal_line=True, include_hue=True)
        
        # SDR depends only on the model, so we drop the hardware grouping (include_hue=False)
        self._plot_metric(df, 'SDR_vocals', "SDR Vocals (Higher is Better)", graph_dir / "SDR_Vocal_Comparison.png", horizontal_line=False, include_hue=False)
        
        # Exactly matching the CSV column 'SDR_instr'
        self._plot_metric(df, 'SDR_instr', "SDR Instrumentals (Higher is Better)", graph_dir / "SDR_Instr_Comparison.png", horizontal_line=False, include_hue=False)
        
        logging.info(f"Reports successfully generated in {self.results_dir}")

    def _plot_metric(self, df: pd.DataFrame, col: str, title: str, path: Path, horizontal_line: bool, include_hue: bool = True):
        """
        Constructs and saves an academically polished bar chart comparing models.
        """
        plt.figure(figsize=(10, 6))
        sns.set_theme(style="whitegrid")
        df['Label'] = df['tool'] + "\n(" + df['model'] + ")"
        
        if include_hue:
            ax = sns.barplot(data=df, x='Label', y=col, hue='device_type', palette="muted", ci=None)
            ax.legend(title="Device Type", frameon=True, fontsize=10)
        else:
            ax = sns.barplot(data=df, x='Label', y=col, palette="muted", ci=None)
            
        if horizontal_line: 
            plt.axhline(1.0, color='red', linestyle='--')
            
        # 1. Academic Headroom: Expand Y-axis by 15% so top labels always have breathing room
        max_val = df[col].max()
        ax.set_ylim(0, max_val * 1.15)
            
        # Bypass strict Pylance type stubs by accessing containers dynamically
        ax_containers = getattr(ax, 'containers', [])
        for container in ax_containers:
            # 2. Placement: Let Matplotlib anchor it automatically to the top edge!
            ax.bar_label(
                container, 
                fmt='%.2f', 
                padding=3,       # Sits 3 points above the bar's top edge
                weight='bold',   # Bold font
                fontsize=11,
                # White text mask to ensure gridlines or red dashed lines don't strike through the text
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5, alpha=0.9)
            ) # type: ignore

        # Academic Typography Polish
        plt.title(title, fontsize=14, pad=15, weight='bold')
        plt.xlabel('Evaluated Models', fontsize=12, labelpad=10)
        
        # 3. Dynamic Y-axis labeling (Works for 'rtf', 'SDR_vocals', 'SDR_instrumental', etc.)
        if col == 'rtf':
            y_label = "Real-Time Factor (RTF)"
        else:
            # Replaces underscores with spaces, capitalizes words, and ensures SDR is uppercase
            clean_col = col.replace('_', ' ').title().replace('Sdr', 'SDR')
            y_label = f"{clean_col} (dB)"
            
        plt.ylabel(y_label, fontsize=12, labelpad=10)

        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio Separation Evaluation Pipeline for Thesis Benchmarking")
    parser.add_argument("--musdb_path", type=str, default="Evaluation/musdb18_test_samples", help="Path to the MUSDB18 dataset folder.")
    parser.add_argument("--model_path", type=str, default="./Models", help="Path to save or read downloaded models.")
    parser.add_argument("--device", type=str, default="AUTO", help="Hardware to evaluate on: 'cpu', 'cuda', or 'auto' (loops both).")
    parser.add_argument("--num_tracks", type=int, default=None, help="Limit evaluation to the first N tracks (useful for fast testing).")
    parser.add_argument("--tool", type=str, default=None, help="Specific tool to test (e.g., 'Demucs', 'Spleeter'). If omitted, runs all.")
    parser.add_argument("--run_separation", type=str, default=None, help="Instantly evaluate a specific tool and model, bypassing settings.json. Format: Tool-Model (e.g., Demucs-htdemucs_ft)")
    parser.add_argument("--flag_download", action="store_true", help="Download models to a temp dir to measure download times.")
    parser.add_argument("--only_reports", action="store_true", help="Skip processing and only generate graphs/LaTeX from the existing CSV.")
    parser.add_argument("--clean_run", action="store_true", help="Delete previous results and output WAVs before starting.")
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging to inspect generated file properties and hardware.")
    
    args = parser.parse_args()
    evaluator = SeparationEvaluator(args)
    if args.only_reports: 
        evaluator.generate_reports(None, from_csv=True, clean_run=args.clean_run)
    else: 
        evaluator.run_evaluation_pipeline()