import os
import sys
import subprocess
import tempfile
import shutil
import re
import logging
import torch
from pydub import AudioSegment

from .utils import setup_ffmpeg_environment, get_unique_filename

class DemucsSeparator:
    def __init__(self):
        setup_ffmpeg_environment()
        logging.info("Demucs Subprocess Wrapper initialized")

    def separate(self, input_path: str, song_name: str, vocals_folder: str, instr_folder: str, model="mdx", channels="Stereo", fmt="wav", sr=44100, bitrate="128k", bit_depth=True, mp3_preset=2, shifts=1, overlap=0.25, device_choice="Auto", flac_compression=5, progress_callback=None):
        try:
            if not os.path.exists(input_path): raise FileNotFoundError("Input file not found.")
            
            target_device = "cpu"
            if device_choice in ["Auto", "GPU"]:
                if torch.cuda.is_available(): target_device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): target_device = "mps"
            prefix = f"[{target_device.upper()}]"

            if progress_callback: progress_callback(10, f"{prefix} Demucs: Preparing...")

            with tempfile.TemporaryDirectory() as temp_dir:
                cmd = [
                    sys.executable, "-m", "demucs.separate",
                    "--two-stems=vocals", "-n", model, "--out", temp_dir,
                    "--shifts", str(shifts), "--overlap", str(overlap), "-d", target_device 
                ]
                
                if fmt == "flac": cmd.append("--flac")
                elif fmt == "mp3": cmd.extend(["--mp3", "--mp3-bitrate", str(bitrate).lower().replace('k', ''), "--mp3-preset", str(mp3_preset)])
                elif fmt == "wav": cmd.append("--int24" if bit_depth else "--float32")
                cmd.append(input_path)

                if progress_callback: progress_callback(20, f"{prefix} Demucs: Starting engine...")

                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)

                total_models = 4 if "mdx" in model.lower() else 1
                current_model_idx, last_percent = 0, 0
                
                # --- NEW: Ring buffer to hold the last 20 lines of console output ---
                debug_log = []

                for line in process.stdout:
                    # Save the line for debugging, keep only the last 20 lines to avoid memory bloat
                    clean_line = line.strip()
                    if clean_line:
                        debug_log.append(clean_line)
                        if len(debug_log) > 20:
                            debug_log.pop(0)

                    match = re.search(r'(\d{1,3})%', line)
                    if match:
                        raw_percent = int(match.group(1))
                        if raw_percent < 10 and last_percent > 80: current_model_idx += 1
                        last_percent = raw_percent
                        safe_idx = min(current_model_idx, total_models - 1)
                        global_percent = ((safe_idx * 100) + raw_percent) / total_models
                        scaled_percent = 20 + (global_percent * 0.65)
                        
                        try:
                            if progress_callback: progress_callback(scaled_percent, f"{prefix} Demucs: Separating (Model {safe_idx + 1}/{total_models}) {raw_percent}%")                        
                        except RuntimeError as e:
                            if str(e) == "ABORT_REQUESTED":
                                process.kill() 
                                raise e 

                process.wait()
                
                # --- NEW: Expose the captured error! ---
                if process.returncode != 0: 
                    error_details = "\n".join(debug_log)
                    logging.error(f"Demucs crashed! Last 20 lines of console output:\n{error_details}")
                    # Passing a short version of the error up to the GUI
                    raise RuntimeError(f"Demucs failed: {error_details[-150:]}") 
                
                if progress_callback: progress_callback(85, f"{prefix} Demucs: Moving files...")

                output_subdir = os.path.join(temp_dir, model, os.path.splitext(os.path.basename(input_path))[0])
                vocals_src, instr_src = os.path.join(output_subdir, f"vocals.{fmt}"), os.path.join(output_subdir, f"no_vocals.{fmt}")

                if not os.path.exists(vocals_src) or not os.path.exists(instr_src): raise FileNotFoundError("Demucs output files missing.")

                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                vocals_dest = get_unique_filename(os.path.join(vocals_folder, f"{song_name}_Demucs_{model}_vocals.{fmt}"))
                instr_dest = get_unique_filename(os.path.join(instr_folder, f"{song_name}_Demucs_{model}_instrumental.{fmt}"))
                
                if progress_callback: progress_callback(90, f"{prefix} Demucs: Processing audio format/channels...")
                
                if channels == "Mono" or ((fmt == "wav" or fmt == "flac") and sr != 44100) or fmt == "flac":
                    v_audio, i_audio = AudioSegment.from_file(vocals_src), AudioSegment.from_file(instr_src)

                    if channels == "Mono":
                        v_audio, i_audio = v_audio.set_channels(1), i_audio.set_channels(1)
                    if sr != 44100:
                        v_audio, i_audio = v_audio.set_frame_rate(sr), i_audio.set_frame_rate(sr)

                    export_kwargs = {"format": fmt}
                    if fmt == "mp3": export_kwargs["bitrate"] = bitrate
                    elif fmt == "flac": export_kwargs["parameters"] = ["-compression_level", str(flac_compression)]

                    v_audio.export(vocals_dest, **export_kwargs)
                    i_audio.export(instr_dest, **export_kwargs)
                else:
                    shutil.move(vocals_src, vocals_dest)
                    shutil.move(instr_src, instr_dest)

                if progress_callback: progress_callback(100, f"{prefix} Demucs: Complete!")
                return True, os.path.basename(vocals_dest), os.path.basename(instr_dest)

        except Exception as e:
            logging.error(f"Demucs error: {e}", exc_info=True)
            return False, None, None