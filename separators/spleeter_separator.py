import os
import sys
import logging
import tempfile
import pathlib
import subprocess
import threading  # Added for non-blocking execution wrapper
import soundfile as sf
import librosa
import music_tag
import numpy as np
from typing import Any
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import time

from .utils import (
    setup_ffmpeg_environment, 
    get_unique_filename, 
    get_audio_metadata, 
    prepare_stem_metadata,
    resolve_tensorflow_device,
    ProgressInterceptor,
    clear_memory_cache
)

class SpleeterSeparator:
    def __init__(self):
        setup_ffmpeg_environment() 
        model_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(model_dir, exist_ok=True)
        os.environ['MODEL_PATH'] = model_dir
        self.model_name = 'spleeter:2stems'

    def separate(self, input_path: str, song_name: str, vocals_folder: str, instr_folder: str, 
                 channels: str = "Stereo", fmt: str = "wav", sr: int = 44100, bitrate: str = "128k", 
                 bit_depth: str = "16-bit", device_choice: str = "Auto", flac_compression: int = 5, 
                 progress_callback: Any = None):
        
        # We will use a mutable container to capture returns from our thread safely
        thread_result = {"success": False, "vocals": None, "instrumental": None}

        def background_worker():
            resolved_device = resolve_tensorflow_device(device_choice)
            root_logger = logging.getLogger()
            handler = ProgressInterceptor(progress_callback, device=resolved_device, tool_name="Spleeter")
            root_logger.addHandler(handler)
            
            process = None  # Reference for the fallback subprocess so we can kill it safely

            try:
                if not os.path.exists(input_path):
                    logging.error(f"Spleeter error: Input file not found at {input_path}")
                    return

                logging.debug(f"Starting Spleeter separation for: '{song_name}'")

                # 1. Metadata setup
                logging.debug("Extracting and preparing audio metadata...")
                original_tags = get_audio_metadata(input_path)
                v_tags = prepare_stem_metadata(original_tags, "Vocals")
                i_tags = prepare_stem_metadata(original_tags, "Instrumental")

                # 2. Heavy Model Loading Moved Completely Inside the Thread Wrapper
                logging.info(f"Spleeter: Initializing AI engine model weights for {song_name}...")
                separator = Separator(self.model_name)
                adapter = AudioAdapter.default()
                logging.debug("Spleeter models loaded into memory successfully.")

                with tempfile.TemporaryDirectory() as temp_dir:
                    clean_input = str(pathlib.Path(input_path).resolve())
                    
                    # 3. Separation
                    try:
                        # Try native execution
                        logging.info("spleeter processing: API active")
                        logging.debug(f"Executing API 'separate_to_file' on: {clean_input}")
                        separator.separate_to_file(clean_input, temp_dir, audio_adapter=adapter, synchronous=True)
                        logging.debug("API separation completed successfully.")
                        
                    except Exception as api_err:
                        # Fallback to Subprocess if API fails
                        logging.warning(f"Spleeter API fallback triggered due to error: {api_err}")
                        logging.info("spleeter processing: fallback active")
                        
                        cmd = [
                            sys.executable, "-m", "spleeter", "separate",
                            "-p", self.model_name, "-o", temp_dir, input_path
                        ]
                        logging.debug(f"Launching subprocess: {' '.join(cmd)}")
                        
                        # Windows specific flag to completely suppress background console windows in compiled EXEs
                        creation_flags = 0x08000000 if os.name == 'nt' else 0
                        
                        # FIX: Merge stderr into stdout so we only read one pipe. 
                        # This prevents the child process from hanging due to a full stdout buffer!
                        process = subprocess.Popen(
                            cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.STDOUT, 
                            text=True,
                            creationflags=creation_flags
                        )
                        
                        # Read lines dynamically to keep the OS pipe buffer empty
                        if process.stdout:
                            for line in process.stdout:
                                logging.debug(f"SPLEETER-SUB: {line.strip()}")
                                
                        process.wait()
                        logging.debug(f"Subprocess exited with return code: {process.returncode}")

                    # 4. Path mapping
                    input_base = os.path.splitext(os.path.basename(input_path))[0]
                    output_folder = os.path.join(temp_dir, input_base)
                    v_src = os.path.join(output_folder, "vocals.wav")
                    i_src = os.path.join(output_folder, "accompaniment.wav")

                    if not os.path.exists(v_src):
                        logging.error(f"Separation failed: Output stems not found in {output_folder}")
                        return
                    
                    logging.debug("Separation successful. Beginning bit-perfect export...")

                    # 5. Bit-Perfect Export Loop
                    stems = [
                        (v_src, f"{song_name}_Spleeter_vocals.{fmt}", v_tags, vocals_folder),
                        (i_src, f"{song_name}_Spleeter_instrumental.{fmt}", i_tags, instr_folder)
                    ]

                    final_paths = []
                    for idx, (src, out_name, tag_dict, folder) in enumerate(stems):
                        save_path = get_unique_filename(os.path.join(folder, out_name))
                        
                        # Use Soundfile for high-res formats
                        if fmt.lower() in ['wav', 'flac']:
                            data, native_sr = sf.read(src, dtype='float32')
                            if native_sr != sr:
                                data = librosa.resample(data.T, orig_sr=native_sr, target_sr=sr).T
                            if channels == "Mono" and data.ndim > 1:
                                data = data.mean(axis=1)

                            st = {"32-bit": "FLOAT", "24-bit": "PCM_24", "16-bit": "PCM_16"}.get(bit_depth, "PCM_16")
                            sf.write(save_path, data, sr, subtype=st)
                        else:
                            from pydub import AudioSegment
                            audio = AudioSegment.from_file(src)
                            if channels == "Mono": audio = audio.set_channels(1)
                            if audio.frame_rate != sr: audio = audio.set_frame_rate(sr)
                            audio.export(save_path, format=fmt, bitrate=bitrate)

                        # Apply tags
                        self._apply_tags(save_path, tag_dict, "Spleeter")
                        final_paths.append(os.path.basename(save_path))

                    # Update mutable dictionary container on success
                    thread_result["success"] = True
                    thread_result["vocals"] = final_paths[0]
                    thread_result["instrumental"] = final_paths[1]
                    logging.debug(f"Export finished successfully for {song_name}.")

            except Exception as e:
                logging.error(f"Spleeter Worker Thread Error on {song_name}: {e}", exc_info=True)
            finally:
                # GUARANTEED CLEANUP: Kill the subprocess if it's still somehow alive
                if process is not None and process.poll() is None:
                    logging.warning("Killing lingering Spleeter fallback subprocess...")
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        
                root_logger.removeHandler(handler)
                logging.debug("Flushing memory and unlinking TensorFlow variables...")
                clear_memory_cache()
                logging.debug(f"Background thread for '{song_name}' closed and destroyed.")

        # Create and initialize the worker thread
        worker_thread = threading.Thread(target=background_worker, daemon=True)
        worker_thread.start()
        
        # Block control return until background thread complete, keeping main Tk loop alive natively
        while worker_thread.is_alive():
            # Crucial: Allows Tkinter window manager to process events/animations/queue loops seamlessly!
            if hasattr(progress_callback, '__self__') and hasattr(progress_callback.__self__, 'update'):
                progress_callback.__self__.update()
            time.sleep(0.05)

        return thread_result["success"], thread_result["vocals"], thread_result["instrumental"]

    def _apply_tags(self, file_path: str, tags: dict, tool: str):
        try:
            f = music_tag.load_file(file_path)
            if f:
                f['title'], f['artist'] = tags.get('title', 'Unknown'), tags.get('artist', 'Unknown')
                f['comment'] = f"Separated by {tool}"
                f.save()
        except Exception as e: 
            logging.warning(f"Tagging failed on {file_path}: {e}")