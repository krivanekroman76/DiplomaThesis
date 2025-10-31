import os
import subprocess
import shlex
import tempfile
import shutil
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
from spleeter.audio import Codec

class SpleeterSeparator:
    def __init__(self):
        self.model = 'spleeter:2stems'
        try:
            self.separator = Separator(self.model)
            print("Spleeter initialized successfully (direct API)")
        except Exception as e:
            print(f"Spleeter init warning: {e} (will use CLI)")

    def _get_unique_filename(self, base_path):
        """Generate a unique filename by appending _1, _2, etc., if the file exists."""
        if not os.path.exists(base_path):
            return base_path
        base, ext = os.path.splitext(base_path)
        counter = 1
        while True:
            new_path = f"{base}_{counter}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def separate(self, input_path, song_name, vocals_folder, instr_folder, fmt="wav", sr=44100, bitrate="128k",):
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"Spleeter: Input file not found: {input_path}")
                return False

            # Map fmt to Codec enum
            codec_map = {
                "wav": Codec.WAV,
                "mp3": Codec.MP3,
                "flac": Codec.FLAC
            }
            codec = codec_map.get(fmt.lower(), Codec.WAV)  # Default to WAV if invalid

            # Create temp dir for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try direct API first with all parameters
                try:
                    self.separator.separate_to_file(audio_descriptor=input_path, destination=temp_dir, audio_adapter=None, codec=codec)
                    print("Spleeter: Direct API separation successful")
                except Exception as api_err:
                    print(f"Spleeter: Direct API failed ({api_err}), falling back to CLI")
                    # CLI fallback with parameters
                    cmd = [
                        'spleeter', 'separate',
                        '-p', self.model,
                        '-o', temp_dir,
                        '--codec', fmt.lower(),
                        '--bitrate', bitrate,
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    if result.returncode != 0:
                        print(f"Spleeter CLI error: {result.stderr}")
                        return False

                # Find and move output files from temp_dir to final folders
                # Assuming filename_format creates files like "song_name/vocals.wav" in temp_dir
                vocals_src = os.path.join(temp_dir, f"{song_name}/vocals.{fmt}")
                instr_src = os.path.join(temp_dir, f"{song_name}/accompaniment.{fmt}")

                if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                    print(f"Spleeter: Output files not found in {temp_dir}. Check filename_format.")
                    return False

                # Ensure final folders exist
                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                # Generate unique destination paths
                ai_suffix = "_S"
                base_vocals_dest = os.path.join(vocals_folder, f"{song_name}{ai_suffix}_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_folder, f"{song_name}{ai_suffix}_instrumental.{fmt}")

                vocals_dest = self._get_unique_filename(base_vocals_dest)
                instr_dest = self._get_unique_filename(base_instr_dest)

                # Move files to unique final locations
                shutil.move(vocals_src, vocals_dest)
                shutil.move(instr_src, instr_dest)

                print(f"Spleeter separation successful for {song_name} in {fmt} format. Files saved as: {vocals_dest}, {instr_dest}")
                return True

        except subprocess.CalledProcessError as e:
            print(f"Spleeter subprocess failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"Spleeter general error: {e}")
            return False