import os
import tempfile
import shutil
import torch
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment  # For format conversion
from openunmix import predict  # High-level API
# Transcription tools
import separators.whisper_transcription as whisper_trans
#import separators.wav2vec2_transcription as wav2vec2_trans 
#import separators.coqui_transcription as coqui_trans 

class OpenUnmixSeparator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            print(f"OpenUnmix: Initializing on {self.device}")
            print("OpenUnmix: Import successful. Models will load on first separation.")
            print(f"OpenUnmix: Ready on {self.device}")
        except Exception as e:
            print(f"OpenUnmix init error: {e}")
            print("OpenUnmix: Check: pip install openunmix-pytorch")
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
                model="umxl", 
                fmt="wav", 
                sr=44100, 
                bitrate="128k", 
                do_transcribe=False, 
                trans_tool="whisper", 
                trans_model="tiny",
                progress_callback=None):  
        """
        Overview: 
            Perform source separation on an audio file using OpenUnmix.
            Saves vocals and accompaniment to specified folders, with optional transcription.
            
        Parameters:
            - input_path (str): Path to the input audio file.
            - song_name (str): Base name for output files (without extension).
            - vocals_folder (str): Folder to save vocal tracks.
            - instr_folder (str): Folder to save instrumental tracks.
            - trans_folder (str): Folder to save transcription files.
            - model (str): OpenUnmix model (e.g., "umxl", "umxhq").
            - fmt (str): Output format ("wav", "mp3", "flac").
            - sr (int): Sample rate (for resampling if needed).
            - bitrate (str): Bitrate for MP3.
            - do_transcribe (bool): Whether to perform transcription.
            - trans_tool (str): Transcription tool ("whisper", etc.).
            - trans_model (str): Model for transcription.
            - progress_callback (callable, optional): Function to call for progress updates (e.g., lambda percent, message: update(percent, message)).
            
        Returns:
            - tuple: (success (bool), vocals_path (str or None), instr_path (str or None), trans_path (str or None)).
        """
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"OpenUnmix: Input file not found: {input_path}")
                return False, None, None, None

            # Call progress callback for initial setup (10%)
            if progress_callback:
                progress_callback(10, "OpenUnmix: Initializing...")

            # Load audio
            audio, original_sr = librosa.load(input_path, sr=44100, mono=False)
            print(f"OpenUnmix: Raw audio shape: {audio.shape}, sr: {original_sr}")

            # Handle mono: Duplicate to stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)
                print(f"OpenUnmix: Fixed audio shape: {audio.shape}")

            # Call progress callback for loading audio (20%)
            if progress_callback:
                progress_callback(20, "OpenUnmix: Loading and preparing audio...")

            with tempfile.TemporaryDirectory() as temp_dir:
                # Call progress callback for running separation (30%)
                if progress_callback:
                    progress_callback(30, "OpenUnmix: Running separation...")

                # Perform separation using predict.separate
                estimates = predict.separate(
                    audio=torch.as_tensor(audio).float(),
                    rate=original_sr,
                    model_str_or_path=model,
                    targets=['vocals'],  # Only vocals
                    residual=True,  # Creates residual for instrumental
                    device=self.device
                )
                print(f"OpenUnmix: Separation complete. Estimates keys: {list(estimates.keys())}")

                # Call progress callback for processing output (60%)
                if progress_callback:
                    progress_callback(60, "OpenUnmix: Processing and saving files...")

                # Extract vocals
                if 'vocals' not in estimates:
                    raise ValueError("No 'vocals' in estimates")
                vocals_raw = estimates['vocals'].detach().cpu().numpy()
                vocals_estimate = self._prepare_audio_for_save(vocals_raw, sr)
                
                # Extract instrumental: Use residual if available, else sum non-vocals
                if 'residual' in estimates:
                    instr_raw = estimates['residual'].detach().cpu().numpy()
                else:
                    non_vocals = [estimates[target].detach().cpu().numpy() for target in estimates if target != 'vocals']
                    if not non_vocals:
                        raise ValueError("No instrumental stems found")
                    instr_raw = np.sum(non_vocals, axis=0)
                instr_estimate = self._prepare_audio_for_save(instr_raw, sr)
                
                # Save temporary WAV files
                vocals_temp_path = os.path.join(temp_dir, 'vocals_temp.wav')
                instr_temp_path = os.path.join(temp_dir, 'instrumental_temp.wav')
                sf.write(vocals_temp_path, vocals_estimate, original_sr)
                sf.write(instr_temp_path, instr_estimate, original_sr)
                
                # Ensure final folders exist
                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                # Generate unique destination paths
                base_vocals_dest = os.path.join(vocals_folder, f"{song_name}_OpenUnmix_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_folder, f"{song_name}_OpenUnmix_instrumental.{fmt}")

                vocals_dest = self._get_unique_filename(base_vocals_dest)
                instr_dest = self._get_unique_filename(base_instr_dest)
                
                # Load and export files
                audio_vocals = AudioSegment.from_wav(vocals_temp_path)
                audio_instr = AudioSegment.from_wav(instr_temp_path)
                if fmt == "mp3":
                    audio_vocals.export(vocals_dest, format="mp3", bitrate=bitrate)
                    audio_instr.export(instr_dest, format="mp3", bitrate=bitrate)
                elif fmt == "flac":
                    audio_vocals.export(vocals_dest, format="flac")
                    audio_instr.export(instr_dest, format="flac")
                else:
                    audio_vocals.export(vocals_dest, format="wav")
                    audio_instr.export(instr_dest, format="wav")
                
                print(f"OpenUnmix separation successful for {song_name} in {fmt} format. Files saved as: {vocals_dest}, {instr_dest}")

                trans_path = ''
                if do_transcribe:
                    # Call progress callback for transcription (70%)
                    if progress_callback:
                        progress_callback(70, "OpenUnmix: Transcribing vocals...")
                    
                    trans_path = os.path.join(trans_folder, f"{song_name}_OpenUnmix_transcription.txt")
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
                        print(f"OpenUnmix: Unknown transcription tool '{trans_tool}'.")
                        
                    if success_trans:
                        print(f"OpenUnmix: Transcription completed for {song_name} by '{trans_tool}' using '{trans_model}'.")
                    else:
                        print(f"OpenUnmix: Transcription failed for {song_name} by '{trans_tool}' using '{trans_model}'.")
                        trans_path = None
                
                # Call progress callback for completion (100%) handled in separation app with returned values
                return True, vocals_dest, instr_dest, trans_path

        except Exception as e:
            print(f"OpenUnmix error: {e}")
            import traceback
            traceback.print_exc()
            return False, None, None, None

    def _prepare_audio_for_save(self, estimate, sr):
        """Helper: Squeeze extra dims, ensure correct shape, and resample if needed."""
        estimate = np.squeeze(estimate)
        print(f"Shape after squeeze: {estimate.shape}")
        
        if estimate.ndim == 2 and estimate.shape[0] < estimate.shape[1]:
            estimate = estimate.T
            print(f"Shape after transpose: {estimate.shape}")
        
        if estimate.ndim == 2 and estimate.shape[1] == 1:
            estimate = estimate[:, 0]
            print(f"Mono flattened to 1D: {estimate.shape}")
        
        if sr != 44100:
            audio_segment = AudioSegment(estimate.tobytes(), frame_rate=44100, sample_width=2, channels=estimate.ndim)
            audio_segment = audio_segment.set_frame_rate(sr)
            estimate = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            if audio_segment.channels == 1:
                estimate = estimate.reshape(-1)
            print(f"Resampled to sr: {sr}, new shape: {estimate.shape}")
        
        return estimate