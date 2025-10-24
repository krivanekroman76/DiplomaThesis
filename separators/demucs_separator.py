import os
import shutil
import sys
from demucs.separate import main as demucs_main
from pydub import AudioSegment  # For format conversion

class DemucsSeparator:
    def __init__(self):
        try:
            from demucs.separate import main
            print("Demucs initialized successfully")
        except ImportError as e:
            raise ImportError(f"Demucs not installed properly: {e}. Run 'pip install demucs'.")

    def separate(self, input_path: str, song_name: str, vocals_dir: str, instr_dir: str, model="mdx", fmt="wav", sr=44100, bitrate=192):
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            print(f"Demucs: Processing input: {input_path}")

            args = [
                "--two-stems=vocals",
                "-n", model,
                input_path
            ]
            print(f"Demucs: Running with args: {args}")

            demucs_main(args)
            print(f"Demucs: Separation completed for {song_name}")

            model_dir = f"separated/{model}"
            output_subdir = os.path.join(model_dir, song_name)
            vocals_src = os.path.join(output_subdir, "vocals.wav")
            instr_src = os.path.join(output_subdir, "no_vocals.wav")

            if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                raise FileNotFoundError(f"Demucs output files not found in {output_subdir}")

            os.makedirs(vocals_dir, exist_ok=True)
            os.makedirs(instr_dir, exist_ok=True)
            ai_suffix = "_D"

            vocals_dest_wav = os.path.join(vocals_dir, f"{song_name}{ai_suffix}_vocals.wav")
            instr_dest_wav = os.path.join(instr_dir, f"{song_name}{ai_suffix}_instrumental.wav")
            shutil.move(vocals_src, vocals_dest_wav)
            shutil.move(instr_src, instr_dest_wav)
            
            if fmt != "wav":
                vocals_dest = os.path.join(vocals_dir, f"{song_name}{ai_suffix}_vocals.{fmt}")
                instr_dest = os.path.join(instr_dir, f"{song_name}{ai_suffix}_instrumental.{fmt}")

                audio_vocals = AudioSegment.from_wav(vocals_dest_wav)
                audio_instr = AudioSegment.from_wav(instr_dest_wav)

                if fmt == "mp3":
                    audio_vocals.export(vocals_dest, format="mp3", bitrate=f"{bitrate}k")
                    audio_instr.export(instr_dest, format="mp3", bitrate=f"{bitrate}k")
                elif fmt == "flac":
                    audio_vocals.set_frame_rate(sr).export(vocals_dest, format="flac")
                    audio_instr.set_frame_rate(sr).export(instr_dest, format="flac")

                os.remove(vocals_dest_wav)
                os.remove(instr_dest_wav)

            print(f"Demucs separation successful for {song_name} in {fmt} format")
            return True

        except Exception as e:
            print(f"Demucs separation error: {str(e)}", file=sys.stderr)
            if os.path.exists("separated"):
                shutil.rmtree("separated", ignore_errors=True)
            return False