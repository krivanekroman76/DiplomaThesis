import os
import shutil
import sys
from demucs.separate import main as demucs_main

class DemucsSeparator:
    def __init__(self):
        # Validate Demucs installation and set defaults
        try:
            from demucs.separate import main  # Test import
            self.default_model = "mdx"  # Reliable 2-stem model (vocals + no_vocals); change to "mdx_extra" if preferred
            print("Demucs initialized successfully")
        except ImportError as e:
            raise ImportError(f"Demucs not installed properly: {e}. Run 'pip install demucs'.")

    def separate(self, input_path: str, song_name: str, suffix: str, vocals_dir: str, instr_dir: str) -> bool:
        """
        Separate audio using Demucs (2-stems: vocals + no_vocals/instrumental).
        Saves .wav files to vocals_dir and instr_dir with suffix (default format).
        Returns True on success, False on failure.
        """
        try:
            # Validate input path
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            print(f"Demucs: Processing input: {input_path}")

            # Build CLI args as a list (avoids shlex/path quoting issues on Windows)
            args = [
                "--two-stems=vocals",  # 2-stem separation (vocals + no_vocals)
                "-n", self.default_model,  # Model name
                input_path  # Full path last (Demucs handles it directly)
            ]
            print(f"Demucs: Running with args: {args}")

            # Run Demucs CLI
            demucs_main(args)
            print(f"Demucs: Separation completed for {song_name}")

            # Expected output paths (Demucs uses 'separated/<model>/<song_name>/')
            model_dir = f"separated/{self.default_model}"
            output_subdir = os.path.join(model_dir, song_name)
            vocals_src = os.path.join(output_subdir, "vocals.wav")
            instr_src = os.path.join(output_subdir, "no_vocals.wav")

            print(f"Demucs: Looking for outputs in {output_subdir}")
            if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                # List dir contents for debugging
                if os.path.exists(output_subdir):
                    print(f"Demucs: Available files in {output_subdir}: {os.listdir(output_subdir)}")
                raise FileNotFoundError(f"Demucs output files not found in {output_subdir}. Check model/args.")

            # Ensure target dirs exist
            os.makedirs(vocals_dir, exist_ok=True)
            os.makedirs(instr_dir, exist_ok=True)

            # Move and rename with suffix
            vocals_dest = os.path.join(vocals_dir, f"{song_name}{suffix}_vocals.wav")
            instr_dest = os.path.join(instr_dir, f"{song_name}{suffix}_instrumental.wav")

            shutil.move(vocals_src, vocals_dest)
            shutil.move(instr_src, instr_dest)

            print(f"Demucs: Files moved to {vocals_dest} and {instr_dest}")

            # Clean up Demucs' fixed output directory
            if os.path.exists("separated"):
                shutil.rmtree("separated", ignore_errors=True)

            return True

        except Exception as e:
            print(f"Demucs separation error: {str(e)}", file=sys.stderr)
            # Clean up on failure
            if os.path.exists("separated"):
                shutil.rmtree("separated", ignore_errors=True)
            return False
