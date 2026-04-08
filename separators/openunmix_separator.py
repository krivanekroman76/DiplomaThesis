import os
import tempfile
import logging
import gc
import torch
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment

from openunmix import predict
from .utils import setup_ffmpeg_environment, get_unique_filename

class OpenUnmixSeparator:
    def __init__(self):
        setup_ffmpeg_environment()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        logging.info(f"OpenUnmix initialized on {self.device}")

    def separate(self, input_path: str, song_name: str, vocals_folder: str, instr_folder: str, model="umxl", channels="Stereo", fmt="wav", sr=44100, bitrate="128k", device_choice="Auto", flac_compression=5, progress_callback=None):  
        try:
            if not os.path.exists(input_path): return False, None, None

            target_device = "cpu"
            if device_choice in ["Auto", "GPU"]:
                if torch.cuda.is_available(): target_device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): target_device = "mps"

            prefix = f"[{target_device.upper()}]"

            if progress_callback: progress_callback(10, f"{prefix} OpenUnmix: Initializing...")

            audio, original_sr = librosa.load(input_path, sr=44100, mono=False)
            if audio.ndim == 1: audio = np.stack([audio, audio], axis=-1)

            if progress_callback: progress_callback(30, f"{prefix} OpenUnmix: Running separation...")

            with tempfile.TemporaryDirectory() as temp_dir:
                estimates = predict.separate(
                        audio=torch.as_tensor(audio).float(),
                        rate=original_sr,
                        model_str_or_path=model,
                        targets=['vocals'], 
                        residual=True, 
                        device=target_device 
                    )

                if progress_callback: progress_callback(60, f"{prefix} OpenUnmix: Processing and saving files...")

                vocals_raw = estimates['vocals'].detach().cpu().numpy()
                vocals_estimate = self._prepare_audio_for_save(vocals_raw, sr)
                
                if 'residual' in estimates:
                    instr_raw = estimates['residual'].detach().cpu().numpy()
                else:
                    non_vocals = [estimates[target].detach().cpu().numpy() for target in estimates if target != 'vocals']
                    instr_raw = np.sum(non_vocals, axis=0)
                instr_estimate = self._prepare_audio_for_save(instr_raw, sr)
                
                vocals_temp_path = os.path.join(temp_dir, 'vocals_temp.wav')
                instr_temp_path = os.path.join(temp_dir, 'instrumental_temp.wav')
                sf.write(vocals_temp_path, vocals_estimate, original_sr)
                sf.write(instr_temp_path, instr_estimate, original_sr)
                
                os.makedirs(vocals_folder, exist_ok=True)
                os.makedirs(instr_folder, exist_ok=True)

                vocals_dest = get_unique_filename(os.path.join(vocals_folder, f"{song_name}_OpenUnmix_{model}_vocals.{fmt}"))
                instr_dest = get_unique_filename(os.path.join(instr_folder, f"{song_name}_OpenUnmix_{model}_instrumental.{fmt}"))
                
                audio_vocals, audio_instr = AudioSegment.from_wav(vocals_temp_path), AudioSegment.from_wav(instr_temp_path)
                
                if channels == "Mono":
                    audio_vocals, audio_instr = audio_vocals.set_channels(1), audio_instr.set_channels(1)
                
                export_kwargs = {"format": fmt}
                if fmt == "mp3": export_kwargs["bitrate"] = bitrate
                elif fmt == "flac": export_kwargs["parameters"] = ["-compression_level", str(flac_compression)]

                audio_vocals.export(vocals_dest, **export_kwargs)
                audio_instr.export(instr_dest, **export_kwargs)
                
                # Cleanup
                del audio_vocals, audio_instr, audio, vocals_raw, instr_raw, estimates
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()

                if progress_callback: progress_callback(100, f"{prefix} OpenUnmix: Complete!")
                
                return True, os.path.basename(vocals_dest), os.path.basename(instr_dest)

        except Exception as e:
            logging.error(f"OpenUnmix error: {e}", exc_info=True)
            return False, None, None

    def _prepare_audio_for_save(self, estimate, sr):
        estimate = np.squeeze(estimate)
        if estimate.ndim == 2 and estimate.shape[0] < estimate.shape[1]: estimate = estimate.T
        if estimate.ndim == 2 and estimate.shape[1] == 1: estimate = estimate[:, 0]
            
        if sr != 44100:
            if estimate.ndim == 2:
                estimate = librosa.resample(y=estimate.T, orig_sr=44100, target_sr=sr).T
            else:
                estimate = librosa.resample(y=estimate, orig_sr=44100, target_sr=sr)
        return estimate