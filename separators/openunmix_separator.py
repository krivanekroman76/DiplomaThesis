import logging
import os
import tempfile
import shutil
import torch
import librosa
import soundfile as sf
import numpy as np
import gc
from pydub import AudioSegment  # For format conversion
from openunmix import predict  # High-level API

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class OpenUnmixSeparator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        try:
            logging.info(f"OpenUnmix: Initializing on {self.device}")
            logging.info("OpenUnmix: Import successful. Models will load on first separation.")
        except Exception as e:
            logging.error(f"OpenUnmix init error: {e}", exc_info=True)

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
                model="umxl", 
                channels="Stereo", 
                fmt="wav", 
                sr=44100, 
                bitrate="128k", 
                device_choice="Auto",
                progress_callback=None):  
        
        try:
            if not os.path.exists(input_path):
                logging.error(f"OpenUnmix: Input file not found: {input_path}")
                return False, None, None

            if progress_callback:
                progress_callback(10, "OpenUnmix: Initializing...")

            # Load audio
            audio, original_sr = librosa.load(input_path, sr=44100, mono=False)
            
            # Handle mono: Duplicate to stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=-1)

            target_device = "cpu"
            if device_choice in ["Auto", "GPU"]:
                if torch.cuda.is_available(): target_device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): target_device = "mps"
            
            prefix = f"[{target_device.upper()}]"
            logging.debug(f"OpenUnmix: Processing on {target_device}")

            # 3. Add the warning:
            if progress_callback:
                progress_callback(30, f"{prefix} OpenUnmix: Separating (Downloading model if first run)...")

            with tempfile.TemporaryDirectory() as temp_dir:
                if progress_callback:
                    progress_callback(30, "OpenUnmix: Running separation (This may take a moment)...")

                # Perform separation
                estimates = predict.separate(
                        audio=torch.as_tensor(audio).float(),
                        rate=original_sr,
                        model_str_or_path=model,
                        targets=['vocals'], 
                        residual=True, 
                        device=target_device # <--- Explicit device here!
                    )

                if progress_callback:
                    progress_callback(60, "OpenUnmix: Processing and saving files...")

                # Extract vocals
                if 'vocals' not in estimates:
                    raise ValueError("No 'vocals' in estimates")
                vocals_raw = estimates['vocals'].detach().cpu().numpy()
                vocals_estimate = self._prepare_audio_for_save(vocals_raw, sr)
                
                # Extract instrumental
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
                base_vocals_dest = os.path.join(vocals_folder, f"{song_name}_OpenUnmix_{model}_vocals.{fmt}")
                base_instr_dest = os.path.join(instr_folder, f"{song_name}_OpenUnmix_{model}_instrumental.{fmt}")

                vocals_dest = self._get_unique_filename(base_vocals_dest)
                instr_dest = self._get_unique_filename(base_instr_dest)
                
                # Load and export files
                audio_vocals = AudioSegment.from_wav(vocals_temp_path)
                audio_instr = AudioSegment.from_wav(instr_temp_path)
                
                if channels == "Mono":
                    audio_vocals = audio_vocals.set_channels(1)
                    audio_instr = audio_instr.set_channels(1)
                
                if fmt == "mp3":
                    audio_vocals.export(vocals_dest, format="mp3", bitrate=bitrate)
                    audio_instr.export(instr_dest, format="mp3", bitrate=bitrate)
                elif fmt == "flac":
                    audio_vocals.export(vocals_dest, format="flac")
                    audio_instr.export(instr_dest, format="flac")
                else:
                    audio_vocals.export(vocals_dest, format="wav")
                    audio_instr.export(instr_dest, format="wav")
                
                logging.info(f"OpenUnmix separation successful for {song_name}")
                
                vocals_name = os.path.basename(vocals_dest)
                instr_name = os.path.basename(instr_dest)

                # --- MEMORY CLEANUP ---
                del audio_vocals
                del audio_instr
                del audio
                del vocals_raw
                del instr_raw
                del estimates
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                # ----------------------

                if progress_callback:
                    progress_callback(100, "OpenUnmix: Complete!")

                return True, vocals_name, instr_name

        except Exception as e:
            logging.error(f"OpenUnmix error: {e}", exc_info=True)
            return False, None, None

    def _prepare_audio_for_save(self, estimate, sr):
        """Helper: Squeeze extra dims, ensure correct shape, and resample if needed."""
        estimate = np.squeeze(estimate)
        
        if estimate.ndim == 2 and estimate.shape[0] < estimate.shape[1]:
            estimate = estimate.T
        
        if estimate.ndim == 2 and estimate.shape[1] == 1:
            estimate = estimate[:, 0]
        
        if sr != 44100:
            audio_segment = AudioSegment(estimate.tobytes(), frame_rate=44100, sample_width=2, channels=estimate.ndim)
            audio_segment = audio_segment.set_frame_rate(sr)
            estimate = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
            if audio_segment.channels == 1:
                estimate = estimate.reshape(-1)
        
        return estimate