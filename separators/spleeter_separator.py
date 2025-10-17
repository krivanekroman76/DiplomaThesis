import os
import subprocess
import shlex
import tempfile
import shutil
from spleeter.separator import Separator
from pydub import AudioSegment  # For format conversion and resampling

class SpleeterSeparator:
    def __init__(self):
        self.model = 'spleeter:2stems'  # Fixed to 2 stems as per your request
        try:
            self.separator = Separator(self.model)
            print("Spleeter initialized successfully (direct API)")
        except Exception as e:
            print(f"Spleeter init warning: {e} (will use CLI)")

    def separate(self, input_path, song_name, ai_suffix, vocals_folder, instr_folder, model="4stems", fmt="wav", sr=44100, bitrate=192, high_quality=False):
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"Spleeter: Input file not found: {input_path}")
                return False

            # Create temp dir
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try direct API first
                try:
                    self.separator.separate_to_file(input_path, temp_dir)
                    print("Spleeter: Direct API separation successful")
                except Exception as api_err:
                    print(f"Spleeter: Direct API failed ({api_err}), falling back to CLI")
                    cmd = [
                        'spleeter', 'separate',
                        '-p', self.model,
                        '-o', temp_dir,
                        shlex.quote(input_path)
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    if result.returncode != 0:
                        print(f"Spleeter CLI error: {result.stderr}")
                        return False

                # Find output files
                base_output = os.path.join(temp_dir, os.path.splitext(os.path.basename(input_path))[0])
                vocals_src = os.path.join(base_output, 'vocals.wav')
                instr_src = os.path.join(base_output, 'accompaniment.wav')

                if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                    print(f"Spleeter: Output files not found in {temp_dir}")
                    return False

                # Load and process vocals
                audio_vocals = AudioSegment.from_wav(vocals_src)
                audio_vocals = audio_vocals.set_frame_rate(sr)  # Resample to specified sr

                # Load and process instrumentals
                audio_instr = AudioSegment.from_wav(instr_src)
                audio_instr = audio_instr.set_frame_rate(sr)  # Resample to specified sr

                # Define destination paths with the specified format
                vocals_dest = os.path.join(vocals_folder, f"{song_name}{ai_suffix}_vocals.{fmt}")
                instr_dest = os.path.join(instr_folder, f"{song_name}{ai_suffix}_instrumental.{fmt}")

                os.makedirs(os.path.dirname(vocals_dest), exist_ok=True)
                os.makedirs(os.path.dirname(instr_dest), exist_ok=True)

                if fmt == "wav":
                    audio_vocals.export(vocals_dest, format="wav")
                    audio_instr.export(instr_dest, format="wav")
                elif fmt == "mp3":
                    audio_vocals.export(vocals_dest, format="mp3", bitrate=f"{bitrate}k")
                    audio_instr.export(instr_dest, format="mp3", bitrate=f"{bitrate}k")
                elif fmt == "flac":
                    audio_vocals.export(vocals_dest, format="flac")
                    audio_instr.export(instr_dest, format="flac")
                else:
                    print(f"Error: Unsupported format '{fmt}'. Defaulting to WAV.")
                    audio_vocals.export(vocals_dest, format="wav")
                    audio_instr.export(instr_dest, format="wav")

                print(f"Spleeter separation successful for {song_name} in {fmt} format")
                return True

        except subprocess.CalledProcessError as e:
            print(f"Spleeter subprocess failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"Spleeter general error: {e}")
            return False