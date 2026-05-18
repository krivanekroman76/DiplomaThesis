import os
import tempfile
import logging
import torch
import librosa
import soundfile as sf
import numpy as np
import time
import music_tag
from typing import Tuple, Optional
from openunmix import predict

from .utils import (
    setup_ffmpeg_environment, 
    get_unique_filename, 
    resolve_torch_device,
    get_audio_metadata, 
    prepare_stem_metadata,
    clear_memory_cache,
    ProgressInterceptor
)

class OpenUnmixSeparator:
    def __init__(self):
        setup_ffmpeg_environment()
        logging.info("OpenUnmix Initialized (Adaptive Hybrid Mode)")

    def separate(self, input_path: str, song_name: str, vocals_folder: str, instr_folder: str, 
                  model="umxhq", channels="Stereo", fmt="wav", sr=44100, bitrate="128k", 
                  bit_depth="16-bit", device_choice="AUTO", flac_compression=5, 
                  progress_callback=None, **kwargs):
        
        target_device = resolve_torch_device(device_choice.upper(), return_string=True)
        root_logger = logging.getLogger()
        handler = ProgressInterceptor(progress_callback, device=str(target_device), tool_name="OpenUnmix")
        root_logger.addHandler(handler)
    
        try:
            if not os.path.exists(input_path): return False, None, None

            # 1. Metadata
            original_tags = get_audio_metadata(input_path)
            v_tags = prepare_stem_metadata(original_tags, "Vocals")
            i_tags = prepare_stem_metadata(original_tags, "Instrumental")
            
            model_to_load = os.path.abspath(model) if os.path.isdir(model) else model

            # 2. Prediction (Returns float32 NumPy arrays)
            try:
                audio_np, _ = librosa.load(input_path, sr=44100, mono=False)
                if audio_np.ndim == 1: audio_np = np.stack([audio_np, audio_np], axis=0)
                v_raw, i_raw = self._predict_core(audio_np, model_to_load, target_device)
            except (RuntimeError, torch.cuda.OutOfMemoryError):
                logging.warning("OpenUnmix OOM. Using Chunking...")
                v_raw, i_raw = self._separate_by_chunks(input_path, model_to_load, target_device, progress_callback)

            # 3. Export Stems
            stems = [
                (v_raw, f"{song_name}_OpenUnmix_{model}_vocals.{fmt}", v_tags, vocals_folder),
                (i_raw, f"{song_name}_OpenUnmix_{model}_instrumental.{fmt}", i_tags, instr_folder)
            ]

            final_filenames = []
            for idx, (data, target_name, tag_dict, out_folder) in enumerate(stems):
                save_path = get_unique_filename(os.path.join(out_folder, target_name))
                
                # --- The Logic: Math is already in float32 ---
                if fmt.lower() in ['wav', 'flac']:
                    # data is (Channels, Samples), Soundfile needs (Samples, Channels)
                    out_data = data.T 
                    if sr != 44100:
                        out_data = librosa.resample(out_data.T, orig_sr=44100, target_sr=sr).T
                    if channels == "Mono" and out_data.ndim > 1:
                        out_data = out_data.mean(axis=1)

                    st = {"32-bit": "FLOAT", "24-bit": "PCM_24", "16-bit": "PCM_16"}.get(bit_depth, "PCM_16")
                    sf.write(save_path, out_data, sr, subtype=st)
                else:
                    # Temporary WAV for Pydub fallback
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, data.T, 44100, subtype='FLOAT')
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(tmp.name)
                        audio.export(save_path, format=fmt, bitrate=bitrate)
                    os.unlink(tmp.name)

                self._apply_tags(save_path, tag_dict, f"OpenUnmix {model}")
                final_filenames.append(os.path.basename(save_path))

            return True, final_filenames[0], final_filenames[1]

        except Exception as e:
            logging.error(f"OpenUnmix Error: {e}", exc_info=True)
            return False, None, None
        finally:
            root_logger.removeHandler(handler)
            clear_memory_cache()

    def _predict_core(self, audio_np, model_str, device) -> Tuple[np.ndarray, np.ndarray]:
        estimates = predict.separate(
            audio=torch.as_tensor(audio_np).float(),
            rate=44100, model_str_or_path=model_str,
            targets=['vocals'], residual=True, device=device
        )
        return (estimates['vocals'].detach().cpu().numpy().squeeze(), 
                estimates['residual'].detach().cpu().numpy().squeeze())

    def _separate_by_chunks(self, input_path, model_str, device, progress_callback):
        """Processes audio in 30s segments with a 200ms NumPy crossfade."""
        audio_np, fs = librosa.load(input_path, sr=44100, mono=False)
        if audio_np.ndim == 1: 
            audio_np = np.stack([audio_np, audio_np], axis=0)

        total_samples = audio_np.shape[1]
        chunk_len = 30 * 44100  # 30 seconds
        fade_len = int(0.2 * 44100) # 200ms fade
        
        v_final = np.zeros_like(audio_np)
        i_final = np.zeros_like(audio_np)

        # Create linear fade arrays (0 to 1 and 1 to 0)
        # Reshaped to (1, fade_len) for easy broadcasting across channels
        fade_in = np.linspace(0, 1, fade_len).reshape(1, -1)
        fade_out = np.linspace(1, 0, fade_len).reshape(1, -1)

        start = 0
        while start < total_samples:
            end = min(start + chunk_len, total_samples)
            chunk = audio_np[:, start:end]

            # Update Progress
            pct = 15 + int((start / total_samples) * 70)
            if progress_callback:
                progress_callback(pct, "OpenUnmix: Processing overlapping chunks...")

            # Predict on the slice
            v_chunk, i_chunk = self._predict_core(chunk, model_str, device)
            
            # --- CROSSFADE LOGIC ---
            if start == 0:
                # First chunk: just place it
                v_final[:, start:end] = v_chunk
                i_final[:, start:end] = i_chunk
            else:
                # Subsequent chunks: fade in the start, blend with the previous end
                # We blend the first 'fade_len' samples of the current chunk 
                # with what is already in the final array
                v_final[:, start:start+fade_len] = (
                    v_final[:, start:start+fade_len] * fade_out + 
                    v_chunk[:, :fade_len] * fade_in
                )
                i_final[:, start:start+fade_len] = (
                    i_final[:, start:start+fade_len] * fade_out + 
                    i_chunk[:, :fade_len] * fade_in
                )
                
                # Place the rest of the chunk (after the fade)
                v_final[:, start+fade_len:end] = v_chunk[:, fade_len:]
                i_final[:, start+fade_len:end] = i_chunk[:, fade_len:]

            # Move the pointer, but subtract fade_len to create the overlap for the next loop
            start += (chunk_len - fade_len)
            
            clear_memory_cache()

        return v_final, i_final

    def _apply_tags(self, file_path: str, tags: dict, tool: str):
        try:
            f = music_tag.load_file(file_path)
            if f:
                f['title'], f['artist'] = tags.get('title', 'Unknown'), tags.get('artist', 'Unknown')
                f['comment'] = f"Separated by {tool}"
                f.save()
        except Exception as e: logging.warning(f"Tagging failed: {e}")