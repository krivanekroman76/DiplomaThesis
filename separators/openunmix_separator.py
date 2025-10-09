import os
import tempfile
import shutil
import torch
import librosa
import soundfile as sf
import numpy as np
from openunmix.predict import separate  # High-level separation (no umx.model needed)

class OpenUnmixSeparator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            print(f"OpenUnmix: Initializing on {self.device}")
            # Simple import test—no manual model loading (separate() handles it)
            print("OpenUnmix: Import successful. Models will load on first separation.")
            print(f"OpenUnmix: Ready on {self.device}")
        except Exception as e:
            print(f"OpenUnmix init error: {e}")
            print("OpenUnmix: Check: pip install openunmix-pytorch")

    def separate(self, input_path, song_name, ai_suffix, vocals_folder, instr_folder):
        try:
            # Check if input exists
            if not os.path.exists(input_path):
                print(f"OpenUnmix: Input file not found: {input_path}")
                return False

            # Load audio (44.1kHz, keep stereo/mono)
            audio, sr = librosa.load(input_path, sr=44100, mono=False)
            print(f"OpenUnmix: Raw audio shape: {audio.shape}, sr: {sr}")

            # Handle mono: Duplicate to stereo (T, 2)
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)  # (T, 2)
            elif audio.ndim == 2 and audio.shape[1] == 1:  # (T, 1) -> (T, 2)
                audio = np.stack([audio[:, 0], audio[:, 0]], axis=-1)

            # Ensure (T, C) shape: Transpose if (C, T)
            if audio.shape[0] < audio.shape[1]:  # e.g., (2, T) -> (T, 2)
                audio = audio.T
            print(f"OpenUnmix: Fixed audio shape: {audio.shape}")

            with tempfile.TemporaryDirectory() as temp_dir:
                # High-level separation: All 4 stems, no 'stems' arg
                estimates = predict.separate(
                    torch.as_tensor(audio).float(),
                    rate=audio.rate,
                    targets=['vocals'],
                    residual=True,
                    device=device,
                )
                print(f"OpenUnmix: Separation complete. Estimates keys: {list(estimates.keys())}")

                # Extract vocals (always present)
                if 'vocals' not in estimates:
                    raise ValueError("No 'vocals' in estimates")
                vocals_estimate = estimates['vocals']  # (T, C) or (C, T)? separate returns (T, C)
                # Ensure (T, C) for sf.write
                if vocals_estimate.shape[0] < vocals_estimate.shape[1]:
                    vocals_estimate = vocals_estimate.T
                vocals_path = os.path.join(temp_dir, 'vocals.wav')
                sf.write(vocals_path, vocals_estimate, sr)

                # Extract instrumental: Use 'other' (non-vocals); fallback to sum of non-vocals if missing
                if 'other' in estimates:
                    instr_estimate = estimates['other']
                else:
                    # Fallback: Combine drums + bass + other (if present)
                    non_vocals = []
                    for stem in ['drums', 'bass', 'other']:
                        if stem in estimates:
                            non_vocals.append(estimates[stem])
                    if non_vocals:
                        instr_estimate = np.sum(non_vocals, axis=0)
                    else:
                        raise ValueError("No non-vocal stems found")
                
                # Ensure (T, C) for instrumental
                if instr_estimate.shape[0] < instr_estimate.shape[1]:
                    instr_estimate = instr_estimate.T
                other_path = os.path.join(temp_dir, 'other.wav')
                sf.write(other_path, instr_estimate, sr)

                # Ensure target dirs
                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                # Move with suffix
                vocals_dest = os.path.join(vocals_folder, f"{song_name}{ai_suffix}_vocals.wav")
                instr_dest = os.path.join(instr_folder, f"{song_name}{ai_suffix}_instrumental.wav")

                shutil.move(vocals_path, vocals_dest)
                shutil.move(other_path, instr_dest)

            print(f"OpenUnmix separation successful for {song_name}")
            return True

        except Exception as e:
            print(f"OpenUnmix error: {e}")
            import traceback
            traceback.print_exc()  # Full stack for debugging
            return False
