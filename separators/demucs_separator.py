import os
import shutil
import sys
import tempfile
from demucs.separate import main as demucs_main
from pydub import AudioSegment  # For fallback WAV resampling if needed

class DemucsSeparator:
    def __init__(self):
        try:
            from demucs.separate import main
            print("Demucs initialized successfully")
        except ImportError as e:
            raise ImportError(f"Demucs not installed properly: {e}. Run 'pip install demucs'.")

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

    def separate( self, input_path: str, song_name: str, vocals_dir: str, instr_dir: str, model="mdx", fmt="wav", sr=44100, bitrate="128k", bit_depth=True, mp3_preset=2, shifts=1):
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            print(f"Demucs: Processing input: {input_path}")

            # Validate fmt and mutually exclusive bit depth options
            supported_fmts = ["wav", "mp3", "flac"]
            if fmt not in supported_fmts:
                raise ValueError(f"Unsupported format '{fmt}'. Supported: {supported_fmts}")
            if bit_depth:
                int24 = True
                float32 = False
            else:
                int24 = True
                float32 = False

            # Create temp dir for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Prepare Demucs arguments
                args = [
                    "--two-stems=vocals",
                    "-n", model,
                    "--out", temp_dir,  # Output to temp dir
                    "--shifts", str(shifts),
                ]
                # Map fmt to Demucs format flags
                if fmt == "flac":
                    args.append("--flac")
                elif fmt == "mp3":
                    args.append("--mp3")
                    args.extend(["--mp3-bitrate", str(bitrate)])
                    args.extend(["--mp3-preset", str(mp3_preset)])
                elif fmt == "wav":
                    if int24:
                        args.append("--int24")
                    elif float32:
                        args.append("--float32")

                args.append(input_path)

                print(f"Demucs: Running with args: {args}")

                demucs_main(args)
                print(f"Demucs: Separation completed for {song_name}")

                # Demucs outputs to a subfolder like "temp_dir/mdx/song_name/"
                model_dir = os.path.join(temp_dir, model)
                output_subdir = os.path.join(model_dir, song_name)
                vocals_src = os.path.join(output_subdir, f"vocals.{fmt}")
                instr_src = os.path.join(output_subdir, f"no_vocals.{fmt}")

                if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                    raise FileNotFoundError(f"Demucs output files not found in {output_subdir}")

                # Ensure final folders exist
                os.makedirs(vocals_dir, exist_ok=True)
                os.makedirs(instr_dir, exist_ok=True)
                ai_suffix = "_D"

                # Generate unique destination paths
                base_vocals_dest = os.path.join(vocals_dir, f"{song_name}{ai_suffix}_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_dir, f"{song_name}{ai_suffix}_instrumental.{fmt}")

                vocals_dest = self._get_unique_filename(base_vocals_dest)
                instr_dest = self._get_unique_filename(base_instr_dest)

                # If output is WAV and resampling is needed, use pydub; otherwise, move directly
                if fmt == "wav" and sr != 44100:  # Assuming Demucs outputs at 44.1kHz
                    audio_vocals = AudioSegment.from_wav(vocals_src)
                    audio_vocals = audio_vocals.set_frame_rate(sr)
                    audio_vocals.export(vocals_dest, format="wav")

                    audio_instr = AudioSegment.from_wav(instr_src)
                    audio_instr = audio_instr.set_frame_rate(sr)
                    audio_instr.export(instr_dest, format="wav")
                else:
                    # Move files directly (Demucs handled format/bitrate/bit depth)
                    shutil.move(vocals_src, vocals_dest)
                    shutil.move(instr_src, instr_dest)

                print(f"Demucs separation successful for {song_name} in {fmt} format. Files saved as: {vocals_dest}, {instr_dest}")
                return True

        except Exception as e:
            print(f"Demucs separation error: {str(e)}", file=sys.stderr)
            return False