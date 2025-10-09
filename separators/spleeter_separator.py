import os
import subprocess
import shlex
import tempfile
import shutil
from spleeter.separator import Separator  # For direct API fallback

class SpleeterSeparator:
    def __init__(self):
        self.model = 'spleeter:2stems'  # Vocals + accompaniment
        try:
            self.separator = Separator(self.model)
            print("Spleeter initialized successfully (direct API)")
        except Exception as e:
            print(f"Spleeter init warning: {e} (will use CLI)")

    def separate(self, input_path, song_name, ai_suffix, vocals_folder, instr_folder):
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"Spleeter: Input file not found: {input_path}")
                return False

            # Create temp dir
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try direct API first (faster, no subprocess issues)
                try:
                    self.separator.separate_to_file(input_path, temp_dir)
                    print("Spleeter: Direct API separation successful")
                except Exception as api_err:
                    print(f"Spleeter: Direct API failed ({api_err}), falling back to CLI")
                    # CLI fallback with quoted path
                    cmd = [
                        'spleeter', 'separate',
                        '-p', self.model,
                        '-o', temp_dir,
                        shlex.quote(input_path)  # Safely quote the path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    if result.returncode != 0:
                        print(f"Spleeter CLI error: {result.stderr}")
                        return False

                # Find and move output files
                base_output = os.path.join(temp_dir, os.path.splitext(os.path.basename(input_path))[0])
                vocals_src = os.path.join(base_output, 'vocals.wav')
                instr_src = os.path.join(base_output, 'accompaniment.wav')

                if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                    print(f"Spleeter: Output files not found in {temp_dir}")
                    return False

                # Move with suffix
                vocals_dest = os.path.join(vocals_folder, f"{song_name}{ai_suffix}_vocals.wav")
                instr_dest = os.path.join(instr_folder, f"{song_name}{ai_suffix}_instrumental.wav")
                
                os.makedirs(os.path.dirname(vocals_dest), exist_ok=True)
                os.makedirs(os.path.dirname(instr_dest), exist_ok=True)
                
                shutil.move(vocals_src, vocals_dest)
                shutil.move(instr_src, instr_dest)

            print(f"Spleeter separation successful for {song_name}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"Spleeter subprocess failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"Spleeter general error: {e}")
            return False
