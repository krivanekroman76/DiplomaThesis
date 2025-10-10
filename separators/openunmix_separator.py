import os
import tempfile
import shutil
import torch
import librosa
import soundfile as sf
import numpy as np
from openunmix import predict  # High-level API from example

class OpenUnmixSeparator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            print(f"OpenUnmix: Initializing on {self.device}")
            # Simple import test—no manual model loading (predict.separate handles it)
            print("OpenUnmix: Import successful. Models will load on first separation.")
            print(f"OpenUnmix: Ready on {self.device}")
        except Exception as e:
            print(f"OpenUnmix init error: {e}")
            print("OpenUnmix: Check: pip install openunmix-pytorch")

    def _prepare_audio_for_save(self, estimate):
        """Helper: Squeeze extra dims and ensure (T, C) or (T,) shape for sf.write."""
        # Squeeze removes 1-sized dims (e.g., (T, C, 1) -> (T, C))
        estimate = np.squeeze(estimate)
        print(f"Shape after squeeze: {estimate.shape}")
        
        # Ensure time-first (T, C); transpose if needed
        if estimate.ndim == 2 and estimate.shape[0] < estimate.shape[1]:  # (C, T) -> (T, C)
            estimate = estimate.T
            print(f"Shape after transpose: {estimate.shape}")
        
        # If mono (1 channel), flatten to 1D (T,)
        if estimate.ndim == 2 and estimate.shape[1] == 1:
            estimate = estimate[:, 0]
            print(f"Mono flattened to 1D: {estimate.shape}")
        
        return estimate

    def separate(self, input_path, song_name, ai_suffix, vocals_folder, instr_folder):
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"OpenUnmix: Input file not found: {input_path}")
                return False

            # Load audio (44.1kHz, keep stereo/mono) - matches example's track.audio
            audio, sr = librosa.load(input_path, sr=44100, mono=False)
            print(f"OpenUnmix: Raw audio shape: {audio.shape}, sr: {sr}")

            # Handle mono: Duplicate to stereo (T, 2) - ensures (T, C) as in example
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)  # (T, 2)
            print(f"OpenUnmix: Fixed audio shape: {audio.shape}")

            with tempfile.TemporaryDirectory() as temp_dir:
                # High-level separation from example: Full 4-stem + residual
                estimates = predict.separate(
                    audio=torch.as_tensor(audio).float(),  # Convert to tensor (T, C)
                    rate=sr,  # Sample rate from Librosa
                    targets=['vocals', 'drums', 'bass', 'other'],  # All stems as in example
                    residual=True,  # Improves reconstruction (as in example)
                    device=self.device
                )
                print(f"OpenUnmix: Separation complete. Estimates keys: {list(estimates.keys())}")

                # Extract vocals (as in example)
                if 'vocals' not in estimates:
                    raise ValueError("No 'vocals' in estimates")
                vocals_raw = estimates['vocals'].detach().cpu().numpy()  # (T, C) or (T, C, 1)
                vocals_estimate = self._prepare_audio_for_save(vocals_raw)
                vocals_path = os.path.join(temp_dir, 'vocals.wav')
                sf.write(vocals_path, vocals_estimate, sr)  # Now 2D/1D safe

                # Extract instrumental/accompaniment: Sum non-vocals (drums + bass + other) as in example
                non_vocals = [
                    estimates[target].detach().cpu().numpy() 
                    for target in estimates 
                    if target != 'vocals'  # Exclude vocals (and residual if present? Wait, include residual for full acc)
                ]
                if not non_vocals:
                    raise ValueError("No non-vocal stems found")
                # Sum along channel axis (example uses [0] for mono, but we keep stereo/multi)
                instr_raw = np.sum(non_vocals, axis=0)  # (T, C) or (T, C, 1)
                instr_estimate = self._prepare_audio_for_save(instr_raw)
                instr_path = os.path.join(temp_dir, 'instrumental.wav')
                sf.write(instr_path, instr_estimate, sr)

                # Ensure target dirs
                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                # Move with suffix
                vocals_dest = os.path.join(vocals_folder, f"{song_name}{ai_suffix}_vocals.wav")
                instr_dest = os.path.join(instr_folder, f"{song_name}{ai_suffix}_instrumental.wav")

                shutil.move(vocals_path, vocals_dest)
                shutil.move(instr_path, instr_dest)

            print(f"OpenUnmix separation successful for {song_name}")
            return True

        except Exception as e:
            print(f"OpenUnmix error: {e}")
            import traceback
            traceback.print_exc()  # Full stack for debugging
            return False
