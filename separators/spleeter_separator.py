import os
import subprocess
import shlex
import tempfile
import shutil
import warnings
from spleeter.separator import Separator
from spleeter.audio import Codec

# Transcription tools
import separators.whisper_transcription as whisper_trans 
#import separators.wav2vec2_transcription as wav2vec2_trans 
#import separators.coqui_transcription as coqui_trans

class SpleeterSeparator:
    """
    Overview: Class for handling audio source separation using Spleeter.
    Supports 2-stem separation (vocals and accompaniment).
    Uses Spleeter's API or CLI fallback, with optional transcription.
    """
    def __init__(self):
        """
        Overview: Initialize the Spleeter separator with model and transcription tools.
        Suppresses TensorFlow/Keras deprecation warnings for cleaner output.
        """
        # Suppress TensorFlow/Keras warnings globally
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
        
        self.model = 'spleeter:2stems'
        try:
            self.separator = Separator(self.model)
            print("Spleeter initialized successfully (direct API)")
        except Exception as e:
            print(f"Spleeter init warning: {e} (will use CLI)")
        self.whisper_trans = whisper_trans.WhisperTranscription()
        #self.wav2vec2_trans = wav2vec2_trans.Wav2Vec2Transcription() 
        #self.coqui_trans = coqui_trans.CoquiTranscription()

    def _get_unique_filename(self, base_path):
        """
        Overview: Generate a unique filename by appending _1, _2, etc., if the file exists.
        
        Parameters:
        - base_path (str): The initial file path to check.
        
        Returns:
        - str: A unique file path that doesn't exist.
        """
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
                fmt="wav", 
                sr=44100, 
                bitrate="128k", 
                do_transcribe=False,  
                trans_tool="whisper", 
                trans_model="tiny",
                progress_callback=None):
        """
        Overview: 
            Perform source separation on an audio file using Spleeter.
            Saves vocals and accompaniment to specified folders, with optional transcription.
        
        Parameters:
            - input_path (str): Path to the input audio file.
            - song_name (str): Base name for output files (without extension).
            - vocals_folder (str): Folder to save vocal tracks.
            - instr_folder (str): Folder to save instrumental tracks.
            - trans_folder (str): Folder to save transcription files.
            - fmt (str): Output format ("wav", "mp3", "flac").
            - sr (int): Sample rate (not used in Spleeter, but kept for consistency).
            - bitrate (str): Bitrate for MP3 (e.g., "128k").
            - do_transcribe (bool): Whether to perform transcription.
            - trans_tool (str): Transcription tool ("whisper", etc.).
            - trans_model (str): Model for transcription (e.g., "tiny").
            - progress_callback (callable, optional): Function to call for progress updates (e.g., lambda percent: update_bar(percent)).
        
        Returns:
            - tuple: (success (bool), vocals_name (str or None), instr_name (str or None), trans_name (str or None)).
        """
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"Spleeter: Input file not found: {input_path}")
                return False, None, None, None

            # Determine codec based on format
            if fmt == "flac":
                codec = Codec.FLAC
            elif fmt == "mp3":
                codec = Codec.MP3
            else:
                codec = Codec.WAV

            # Call progress callback for initial loading (10%)
            if progress_callback:
                progress_callback(10, "Loading Spleeter model...")

            # Create temp dir for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Try direct API first
                try:
                    self.separator.separate_to_file(audio_descriptor=input_path, destination=temp_dir, audio_adapter=None, codec=codec)
                    print("Spleeter: Direct API separation successful")
                    if progress_callback:
                        progress_callback(30, "Spleeter separation in progress...")
                except Exception as api_err:
                    print(f"Spleeter: Direct API failed ({api_err}), falling back to CLI")
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
                        return False, None, None, None
                    if progress_callback:
                        progress_callback(30, "Spleeter CLI separation in progress...")

                vocals_src = os.path.join(temp_dir, f"{song_name}/vocals.{fmt}")
                instr_src = os.path.join(temp_dir, f"{song_name}/accompaniment.{fmt}")
                
                if not os.path.exists(vocals_src) or not os.path.exists(instr_src):
                    print(f"Spleeter: Output files not found in {temp_dir}. Check filename_format.")
                    return False, None, None, None

                # Call progress callback for file moving (60%)
                if progress_callback:
                    progress_callback(60, "Moving files to output folders...")

                # Ensure final folders exist
                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                base_vocals_dest = os.path.join(vocals_folder, f"{song_name}_Spleeter_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_folder, f"{song_name}_Spleeter_instrumental.{fmt}")

                vocals_dest = self._get_unique_filename(base_vocals_dest)
                instr_dest = self._get_unique_filename(base_instr_dest)

                # Move files to unique final locations
                shutil.move(vocals_src, vocals_dest)
                shutil.move(instr_src, instr_dest)

                print(f"Spleeter separation successful for {song_name} in {fmt} format. Files saved as: {vocals_dest}, {instr_dest}")
                
            # Handle transcription if enabled (runs synchronously after separation)
            trans_name = None
            if do_transcribe:
                if progress_callback:
                    progress_callback(70, "Transcribing vocals...")
                trans_path = os.path.join(trans_folder, f"{song_name}_Spleeter_transcription.txt")
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
                    print(f"Spleeter: Unknown transcription tool '{trans_tool}'.")
                    
                if success_trans:
                    print(f"Spleeter: Transcription completed for {song_name} by '{trans_tool}' using '{trans_model}'.")
                    trans_name = os.path.basename(trans_path)  # Return file name only
                    if progress_callback:
                        progress_callback(90, "Transcribing vocals done!")
                else:
                    print(f"Spleeter: Transcription failed for {song_name} by '{trans_tool}' using '{trans_model}'.")
                    trans_name = None  # Ensure it's None on failure

            # Call progress callback for completion (100%)
            if progress_callback:
                progress_callback(100, "Separation done!")
            
            # Return file names (not paths) for GUI
            vocals_name = os.path.basename(vocals_dest) if vocals_dest else None
            instr_name = os.path.basename(instr_dest) if instr_dest else None
            return True, vocals_name, instr_name, trans_name

        except subprocess.CalledProcessError as e:
            print(f"Spleeter subprocess failed: {e.stderr}")
            return False, None, None, None
        except Exception as e:
            print(f"Spleeter general error: {e}")
            return False, None, None, None