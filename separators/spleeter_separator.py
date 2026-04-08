import os
import subprocess
import tempfile
import shutil
import logging
from pydub import AudioSegment

from spleeter.separator import Separator
from spleeter.audio import Codec
from .utils import setup_ffmpeg_environment, get_unique_filename

class SpleeterSeparator:
    def __init__(self):
        setup_ffmpeg_environment() # Ensure FFmpeg is ready
        self.model = 'spleeter:2stems'
        try:
            self.separator = Separator(self.model)
            logging.info("Spleeter initialized successfully (direct API)")
        except Exception as e:
            logging.error(f"Spleeter init warning: {e} (will use CLI)", exc_info=True)

    def separate(self, input_path, song_name, vocals_folder, instr_folder, channels="Stereo", fmt="wav", sr=44100, bitrate="128k", device_choice="Auto", flac_compression=5, progress_callback=None):
        try:
            if not os.path.exists(input_path): return False, None, None

            # --- Create a prefix based on device_choice ---
            resolved_device = "GPU" if device_choice in ["Auto", "GPU"] else "CPU"
            prefix = f"[{resolved_device}]"

            codec = Codec.FLAC if fmt == "flac" else Codec.MP3 if fmt == "mp3" else Codec.WAV
            if progress_callback: progress_callback(10, f"{prefix} Spleeter: Loading & Preparing Audio...")

            with tempfile.TemporaryDirectory() as temp_dir:
                try: 
                    if progress_callback: progress_callback(20, f"{prefix} Spleeter: Separating audio... (This may take a moment)")
                    self.separator.separate_to_file(audio_descriptor=input_path, destination=temp_dir, audio_adapter=None, codec=codec)
                    if progress_callback: progress_callback(70, f"{prefix} Spleeter: Separation complete! Saving temporary files...")
                except Exception as api_err:
                    logging.info(f"Spleeter API failed, falling back to CLI. Error: {api_err}")
                    cli_env = os.environ.copy()
                    
                    if device_choice == "CPU": cli_env["CUDA_VISIBLE_DEVICES"] = "-1"
                    elif device_choice == "GPU" and "CUDA_VISIBLE_DEVICES" in cli_env and cli_env["CUDA_VISIBLE_DEVICES"] == "-1": del cli_env["CUDA_VISIBLE_DEVICES"]

                    cmd = ['spleeter', 'separate', '-p', self.model, '-o', temp_dir, '--codec', fmt.lower(), '--bitrate', bitrate]
                    if progress_callback: progress_callback(20, f"{prefix} Spleeter: Separating audio via CLI...")
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True, env=cli_env)
                    if result.returncode != 0: return False, None, None
                    if progress_callback: progress_callback(70, f"{prefix} Spleeter: Separation complete!")

                vocals_src = os.path.join(temp_dir, f"{song_name}/vocals.{fmt}")
                instr_src = os.path.join(temp_dir, f"{song_name}/accompaniment.{fmt}")
                if not os.path.exists(vocals_src) or not os.path.exists(instr_src): return False, None, None

                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                base_vocals_dest = os.path.join(vocals_folder, f"{song_name}_Spleeter_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_folder, f"{song_name}_Spleeter_instrumental.{fmt}")

                # Using the imported helper function!
                vocals_dest = get_unique_filename(base_vocals_dest)
                instr_dest = get_unique_filename(base_instr_dest)

                if progress_callback: progress_callback(80, f"{prefix} Spleeter: Processing final audio format...")

                if channels == "Mono" or fmt == "flac":
                    v_audio = AudioSegment.from_file(vocals_src)
                    i_audio = AudioSegment.from_file(instr_src)
                    if channels == "Mono":
                        v_audio = v_audio.set_channels(1)
                        i_audio = i_audio.set_channels(1)
                    
                    export_kwargs = {"format": fmt}
                    if fmt == "mp3": export_kwargs["bitrate"] = bitrate
                    elif fmt == "flac": export_kwargs["parameters"] = ["-compression_level", str(flac_compression)]

                    v_audio.export(vocals_dest, **export_kwargs)
                    i_audio.export(instr_dest, **export_kwargs)
                else:
                    shutil.move(vocals_src, vocals_dest)
                    shutil.move(instr_src, instr_dest)

            if progress_callback: progress_callback(100, f"{prefix} Spleeter: Separation done!")
            return True, os.path.basename(vocals_dest), os.path.basename(instr_dest)
        except Exception as e:
            logging.error(f"Spleeter error: {e}", exc_info=True)
            return False, None, None