import os
import shutil
import sys
import tempfile
from demucs.separate import main as demucs_main
from pydub import AudioSegment  # For fallback WAV resampling if needed
# Transcription tools
import separators.whisper_transcription as whisper_trans
#import separators.wav2vec2_transcription as wav2vec2_trans 
#import separators.coqui_transcription as coqui_trans 

class DemucsSeparator:
    def __init__(self):
        try:
            from demucs.separate import main
            print("Demucs initialized successfully")
        except ImportError as e:
            raise ImportError(f"Demucs not installed properly: {e}. Run 'pip install demucs'.")
        self.whisper_trans = whisper_trans.WhisperTranscription()
        #self.wav2vec2_trans = wav2vec2_trans.Wav2Vec2Transcription() 
        #self.coqui_trans = coqui_trans.CoquiTranscription()

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

    def separate(self, 
                input_path: str, 
                song_name: str, 
                vocals_folder: str, 
                instr_folder: str,
                trans_folder: str, 
                model="mdx", 
                fmt="wav", 
                sr=44100, 
                bitrate="128k", 
                bit_depth=True, 
                mp3_preset=2, 
                shifts=1, 
                do_transcribe=False,
                trans_tool="whisper", 
                trans_model="tiny",
                progress_callback=None):
        """
        Overview: 
            Perform source separation on an audio file using Demucs.
            Saves vocals and accompaniment to specified folders, with optional transcription.
            
        Parameters:
            - input_path (str): Path to the input audio file.
            - song_name (str): Base name for output files (without extension).
            - vocals_folder (str): Folder to save vocal tracks.
            - instr_folder (str): Folder to save instrumental tracks.
            - trans_folder (str): Folder to save transcription files.
            - model (str): Demucs model (e.g., "mdx", "htdemucs").
            - fmt (str): Output format ("wav", "mp3", "flac").
            - sr (int): Sample rate (for resampling if needed).
            - bitrate (str): Bitrate for MP3.
            - bit_depth (bool): True for 24-bit WAV, False for float32.
            - mp3_preset (int): MP3 preset (2-7).
            - shifts (int): Number of shifts for quality.
            - do_transcribe (bool): Whether to perform transcription.
            - trans_tool (str): Transcription tool ("whisper", etc.).
            - trans_model (str): Model for transcription.
            - progress_callback (callable, optional): Function to call for progress updates (e.g., lambda percent, message: update(percent, message)).
            
        Returns:
            - tuple: (success (bool), vocals_path (str or None), instr_path (str or None)).
        """
        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Input file not found: {input_path}")
            print(f"Demucs: Processing input: {input_path}")

            # Call progress callback for initial setup (10%)
            if progress_callback:
                progress_callback(10, "Demucs: Initializing...")

            # Validate fmt and mutually exclusive bit depth options
            supported_fmts = ["wav", "mp3", "flac"]
            if fmt not in supported_fmts:
                raise ValueError(f"Unsupported format '{fmt}'. Supported: {supported_fmts}")
            if bit_depth:
                int24 = True
                float32 = False
            else:
                float32 = False
                int24 = True

            # Call progress callback for preparation (20%)
            if progress_callback:
                progress_callback(20, "Demucs: Preparing arguments...")

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

                # Call progress callback for running Demucs (30%)
                if progress_callback:
                    progress_callback(30, "Demucs: Running separation...")

                demucs_main(args)
                print(f"Demucs: Separation completed for {song_name}")

                # Call progress callback for post-separation (60%)
                if progress_callback:
                    progress_callback(60, "Demucs: Processing output files...")

                # Demucs outputs to a subfolder like "temp_dir/mdx/song_name/"
                model_dir = os.path.join(temp_dir, model)
                input_stem = os.path.splitext(os.path.basename(input_path))[0]
                output_subdir = os.path.join(model_dir, input_stem)
                vocals_src = os.path.join(output_subdir, f"vocals.{fmt}")
                instr_src = os.path.join(output_subdir, f"no_vocals.{fmt}")


                if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                    raise FileNotFoundError(f"Demucs output files not found in {output_subdir}")

                # Ensure final folders exist
                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                # Generate unique destination paths
                base_vocals_dest = os.path.join(vocals_folder, f"{song_name}_Demucs_{model}_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_folder, f"{song_name}_Demucs_{model}_instrumental.{fmt}")

                vocals_dest = self._get_unique_filename(base_vocals_dest)
                instr_dest = self._get_unique_filename(base_instr_dest)

                # If output is WAV or FLAC and resampling is needed, use pydub; otherwise, move directly
                if (fmt == "wav" or fmt == "flac") and sr != 44100:  # Assuming Demucs outputs at 44.1kHz
                    audio_vocals = AudioSegment.from_wav(vocals_src)
                    audio_vocals = audio_vocals.set_frame_rate(sr)
                    audio_vocals.export(vocals_dest, format=fmt)

                    audio_instr = AudioSegment.from_wav(instr_src)
                    audio_instr = audio_instr.set_frame_rate(sr)
                    audio_instr.export(instr_dest, format=fmt)
                else:
                    # Move files directly (Demucs handled format/bitrate/bit depth)
                    shutil.move(vocals_src, vocals_dest)
                    shutil.move(instr_src, instr_dest)

                print(f"Demucs separation successful for {song_name} in {fmt} format. Files saved as: {vocals_dest}, {instr_dest}")
                
                trans_name = None
                if do_transcribe:
                    # Call progress callback for transcription (70%)
                    if progress_callback:
                        progress_callback(70, "Demucs: Transcribing vocals...")
                    
                    trans_path = os.path.join(trans_folder, f"{song_name}_Demucs_transcription.txt")
                    success_trans = False
                    if trans_tool == "whisper":
                        success_trans = self.whisper_trans.transcribe(vocals_dest, trans_path, trans_model)
                    elif trans_tool == "wav2vec2":
                        print("Placeholder for wav2vec2 transcription tool")
                        #success_trans = self.wav2vec2_trans.transcribe(vocals_dest, trans_path, trans_model)
                    elif trans_tool == "coqui":
                        print("Placeholder for coqui transcription tool")
                        #success_trans = self.coqui_trans.transcribe(vocals_dest, trans_path, trans_model)
                    else:
                        print(f"Demucs: Unknown transcription tool '{trans_tool}'.")
                        
                    if success_trans:
                        print(f"Demucs: Transcription completed for {song_name} by '{trans_tool}' using '{trans_model}'.")
                        trans_name = os.path.basename(trans_path)  # Return file name only
                        if progress_callback:
                            progress_callback(90, "Transcribing vocals done!")
                    else:
                        print(f"Demucs: Transcription failed for {song_name} by '{trans_tool}' using '{trans_model}'.")
                        trans_path = None

                # Return file names (not paths) for GUI
                vocals_name = os.path.basename(vocals_dest) if vocals_dest else None
                instr_name = os.path.basename(instr_dest) if instr_dest else None
                return True, vocals_name, instr_name, trans_name

        except Exception as e:
            print(f"Demucs separation error: {str(e)}", file=sys.stderr)
            return False, None, None, None