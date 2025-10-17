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
            # Simple import test—models will load on first separation via predict.separate
            print("OpenUnmix: Import successful. Models will load on first separation.")
            print(f"OpenUnmix: Ready on {self.device} with residual enabled by default.")
        except Exception as e:
            print(f"OpenUnmix init error: {e}")
            print("OpenUnmix: Check: pip install openunmix-pytorch")

    def _prepare_audio_for_save(self, estimate, sr):
        """Helper: Squeeze extra dims, ensure correct shape, and resample if needed."""
        estimate = np.squeeze(estimate)  # Remove 1-sized dims
        print(f"Shape after squeeze: {estimate.shape}")
        
        # Ensure time-first (T, C); transpose if needed
        if estimate.ndim == 2 and estimate.shape[0] < estimate.shape[1]:  # (C, T) -> (T, C)
            estimate = estimate.T
            print(f"Shape after transpose: {estimate.shape}")
        
        # If mono, flatten to 1D
        if estimate.ndim == 2 and estimate.shape[1] == 1:
            estimate = estimate[:, 0]
            print(f"Mono flattened to 1D: {estimate.shape}")
        
        # Resample if specified sr differs from original
        if sr != 44100:  # Assuming original is 44100; adjust if needed
            audio_segment = AudioSegment(estimate.tobytes(), frame_rate=44100, sample_width=2, channels=estimate.ndim)
            audio_segment = audio_segment.set_frame_rate(sr)
            estimate = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            if audio_segment.channels == 1:
                estimate = estimate.reshape(-1)  # Ensure 1D for mono
            print(f"Resampled to sr: {sr}, new shape: {estimate.shape}")
        
        return estimate

    def separate(self, input_path, song_name, ai_suffix, vocals_folder, instr_folder, model, fmt, sr, bitrate, high_quality):
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"OpenUnmix: Input file not found: {input_path}")
                return False

            # Load audio
            audio, original_sr = librosa.load(input_path, sr=44100, mono=False)  # Load at 44100 Hz
            print(f"OpenUnmix: Raw audio shape: {audio.shape}, sr: {original_sr}")

            # Handle mono: Duplicate to stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)  # (T, 2)
            print(f"OpenUnmix: Fixed audio shape: {audio.shape}")

            with tempfile.TemporaryDirectory() as temp_dir:
                # Perform separation with residual enabled
                estimates = predict.separate(
                    audio=torch.as_tensor(audio).float(),  # Convert to tensor
                    rate=original_sr,  # Use original sample rate
                    targets=['vocals', 'drums', 'bass', 'other'],  # Full stems as in example
                    residual=True,  # Ensures it works with the debugged logic
                    device=self.device
                )
                print(f"OpenUnmix: Separation complete. Estimates keys: {list(estimates.keys())}")

                # Extract vocals
                if 'vocals' not in estimates:
                    raise ValueError("No 'vocals' in estimates")
                vocals_raw = estimates['vocals'].detach().cpu().numpy()  # (T, C)
                vocals_estimate = self._prepare_audio_for_save(vocals_raw, sr)
                
                # Extract instrumental: Sum non-vocals
                non_vocals = [estimates[target].detach().cpu().numpy() for target in estimates if target != 'vocals']
                if not non_vocals:
                    raise ValueError("No non-vocal stems found")
                instr_raw = np.sum(non_vocals, axis=0)  # Sum across stems
                instr_estimate = self._prepare_audio_for_save(instr_raw, sr)
                
                # Save temporary WAV files for conversion
                vocals_temp_path = os.path.join(temp_dir, 'vocals_temp.wav')
                instr_temp_path = os.path.join(temp_dir, 'instrumental_temp.wav')
                sf.write(vocals_temp_path, vocals_estimate, original_sr)  # Save as WAV first
                sf.write(instr_temp_path, instr_estimate, original_sr)
                
                # Convert to user-defined format using pydub
                vocals_dest = os.path.join(vocals_folder, f"{song_name}{ai_suffix}_vocals.{fmt}")
                instr_dest = os.path.join(instr_folder, f"{song_name}{ai_suffix}_instrumental.{fmt}")
                
                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)
                
                # Load and export vocals
                audio_vocals = AudioSegment.from_wav(vocals_temp_path)
                if fmt == "wav":
                    audio_vocals.export(vocals_dest, format="wav")
                elif fmt == "mp3":
                    audio_vocals.export(vocals_dest, format="mp3", bitrate=f"{bitrate}k")
                elif fmt == "flac":
                    audio_vocals.export(vocals_dest, format="flac")
                else:
                    print(f"Error: Unsupported format '{fmt}'. Defaulting to WAV.")
                    audio_vocals.export(vocals_dest, format="wav")
                
                # Load and export instrumentals
                audio_instr = AudioSegment.from_wav(instr_temp_path)
                if fmt == "wav":
                    audio_instr.export(instr_dest, format="wav")
                elif fmt == "mp3":
                    audio_instr.export(instr_dest, format="mp3", bitrate=f"{bitrate}k")
                elif fmt == "flac":
                    audio_instr.export(instr_dest, format="flac")
                else:
                    audio_instr.export(instr_dest, format="wav")  # Default
                
                print(f"OpenUnmix separation successful for {song_name} in {fmt} format")
                return True

        except Exception as e:
            print(f"OpenUnmix error: {e}")
            import traceback
            traceback.print_exc()  # Full stack for debugging
            return False