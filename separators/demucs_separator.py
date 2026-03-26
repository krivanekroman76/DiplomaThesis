import logging
import os
import shutil
import tempfile
import subprocess
import sys
import re
import torch
from pydub import AudioSegment

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

class DemucsSeparator:
    def __init__(self):
        logging.info("Demucs Subprocess Wrapper initialized")

    def _get_unique_filename(self, base_path):
        if not os.path.exists(base_path):
            return base_path
        base, ext = os.path.splitext(base_path)
        counter = 1
        while True:
            new_path = f"{base}_{counter}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def separate(self, 
                 input_path: str, 
                 song_name: str, 
                 vocals_folder: str, 
                 instr_folder: str, 
                 model="mdx",
                 channels="Stereo", 
                 fmt="wav", 
                 sr=44100, 
                 bitrate="128k", 
                 bit_depth=True, 
                 mp3_preset=2, 
                 shifts=1,
                 device_choice="Auto", 
                 progress_callback=None):
        
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            target_device = "cpu"
            if device_choice in ["Auto", "GPU"]:
                if torch.cuda.is_available(): target_device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): target_device = "mps"
            
            prefix = f"[{target_device.upper()}]"
            logging.debug(f"Demucs Subprocess: Routing to {target_device}")

            if progress_callback: progress_callback(10, f"{prefix} Demucs: Preparing (Downloading models if needed)...")

            with tempfile.TemporaryDirectory() as temp_dir:
                cmd = [
                    sys.executable, "-m", "demucs.separate",
                    "--two-stems=vocals",
                    "-n", model,
                    "--out", temp_dir,
                    "--shifts", str(shifts),
                    "-d", target_device  # <--- NEW: Forces Demucs to use specific hardware
                ]
                
                if fmt == "flac":
                    cmd.append("--flac")
                elif fmt == "mp3":
                    cmd.extend(["--mp3", "--mp3-bitrate", str(bitrate), "--mp3-preset", str(mp3_preset)])
                elif fmt == "wav":
                    cmd.append("--int24" if bit_depth else "--float32")

                cmd.append(input_path)

                if progress_callback: progress_callback(20, "Demucs: Starting engine...")

                # 2. Run as a Subprocess
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, 
                    text=True,
                    bufsize=1, 
                    universal_newlines=True 
                )

                # --- GLOBAL PROGRESS TRACKING FOR ENSEMBLES ---
                # MDX models use 4 sub-models. HTDemucs usually uses 1.
                total_models = 4 if "mdx" in model.lower() else 1
                current_model_idx = 0
                last_percent = 0

                # 3. Read the terminal output in real-time
                for line in process.stdout:
                    match = re.search(r'(\d{1,3})%', line)
                    if match:
                        raw_percent = int(match.group(1))

                        # If we drop from ~100% back to ~0%, a new sub-model has started!
                        if raw_percent < 10 and last_percent > 80:
                            current_model_idx += 1
                        
                        last_percent = raw_percent
                        safe_model_idx = min(current_model_idx, total_models - 1)

                        # Calculate global completion (0 to 100 across all models)
                        global_raw_percent = ((safe_model_idx * 100) + raw_percent) / total_models
                        
                        # Scale to fit the UI phase (20% to 85%)
                        scaled_percent = 20 + (global_raw_percent * 0.65)
                        
                        try:
                            if progress_callback:
                                        progress_callback(scaled_percent, f"{prefix} Demucs: Separating (Model {safe_model_idx + 1}/{total_models}) {raw_percent}%")
                        except RuntimeError as e:
                            if str(e) == "ABORT_REQUESTED":
                                process.kill() # Kills the Demucs zombie!
                                raise e 

                # Wait for the process to actually close
                process.wait()

                if process.returncode != 0:
                    raise RuntimeError("Demucs Subprocess failed. Check terminal logs.")

                # 4. Handle output files
                if progress_callback: progress_callback(85, "Demucs: Moving files...")

                model_dir = os.path.join(temp_dir, model)
                input_stem = os.path.splitext(os.path.basename(input_path))[0]
                output_subdir = os.path.join(model_dir, input_stem)
                
                vocals_src = os.path.join(output_subdir, f"vocals.{fmt}")
                instr_src = os.path.join(output_subdir, f"no_vocals.{fmt}")

                if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                    raise FileNotFoundError("Demucs output files missing.")

                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                base_vocals_dest = os.path.join(vocals_folder, f"{song_name}_Demucs_{model}_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_folder, f"{song_name}_Demucs_{model}_instrumental.{fmt}")

                vocals_dest = self._get_unique_filename(base_vocals_dest)
                instr_dest = self._get_unique_filename(base_instr_dest)
                
                # Handling output files
                if progress_callback: progress_callback(90, "Demucs: Processing audio format/channels...")
                
                # If we need to change Sample Rate OR convert to Mono, use PyDub
                if channels == "Mono" or ((fmt == "wav" or fmt == "flac") and sr != 44100):
                    v_audio = AudioSegment.from_file(vocals_src)
                    i_audio = AudioSegment.from_file(instr_src)

                    # Downmix to Mono
                    if channels == "Mono":
                        v_audio = v_audio.set_channels(1)
                        i_audio = i_audio.set_channels(1)
                    
                    # Resample
                    if sr != 44100:
                        v_audio = v_audio.set_frame_rate(sr)
                        i_audio = i_audio.set_frame_rate(sr)

                    # Export with correct settings
                    export_kwargs = {"format": fmt}
                    if fmt == "mp3":
                        export_kwargs["bitrate"] = bitrate

                    v_audio.export(vocals_dest, **export_kwargs)
                    i_audio.export(instr_dest, **export_kwargs)
                else:
                    # If Stereo and standard sample rate, just move the files (faster)
                    shutil.move(vocals_src, vocals_dest)
                    shutil.move(instr_src, instr_dest)

                if progress_callback: progress_callback(100, "Demucs: Complete!")
                return True, os.path.basename(vocals_dest), os.path.basename(instr_dest)

        except Exception as e:
            logging.error(f"Demucs separation error: {str(e)}", exc_info=True)
            return False, None, None