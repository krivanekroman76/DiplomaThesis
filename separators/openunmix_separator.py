import os
import tempfile
import shutil
import torch
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment  # For format conversion
from openunmix import predict  # High-level API

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

    def separate(self, input_path, song_name, vocals_folder, instr_folder, model="umxl", fmt="wav", sr=44100, bitrate=192):
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"OpenUnmix: Input file not found: {input_path}")
                return False

            # Load audio
            audio, original_sr = librosa.load(input_path, sr=44100, mono=False)
            print(f"OpenUnmix: Raw audio shape: {audio.shape}, sr: {original_sr}")

            # Handle mono: Duplicate to stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)
            print(f"OpenUnmix: Fixed audio shape: {audio.shape}")

            with tempfile.TemporaryDirectory() as temp_dir:
                # Perform separation using predict.separate (simplified as per example)
                estimates = predict.separate(
                    audio=torch.as_tensor(audio).float(),
                    rate=original_sr,
                    model_str_or_path=model,
                    targets=['vocals'],  # Only vocals
                    residual=True,  # Creates residual for instrumental
                    device=self.device
                )
                print(f"OpenUnmix: Separation complete. Estimates keys: {list(estimates.keys())}")

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
                ai_suffix = "_O"

                # Generate unique destination paths
                base_vocals_dest = os.path.join(vocals_folder, f"{song_name}{ai_suffix}_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_folder, f"{song_name}{ai_suffix}_instrumental.{fmt}")

                vocals_dest = self._get_unique_filename(base_vocals_dest)
                instr_dest = self._get_unique_filename(base_instr_dest)
                
                # Load and export files
                audio_vocals = AudioSegment.from_wav(vocals_temp_path)
                audio_instr = AudioSegment.from_wav(instr_temp_path)
                if fmt == "mp3":
                    audio_vocals.export(vocals_dest, format="mp3", bitrate=f"{bitrate}k")
                    audio_instr.export(instr_dest, format="mp3", bitrate=f"{bitrate}k")
                elif fmt == "flac":
                    audio_vocals.export(vocals_dest, format="flac")
                    audio_instr.export(instr_dest, format="flac")
                else:
                    audio_vocals.export(vocals_dest, format="wav")
                    audio_instr.export(instr_dest, format="wav")
                
                print(f"OpenUnmix separation successful for {song_name} in {fmt} format. Files saved as: {vocals_dest}, {instr_dest}")
                return True

        except Exception as e:
            print(f"OpenUnmix error: {e}")
            import traceback
            traceback.print_exc()
            return False

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