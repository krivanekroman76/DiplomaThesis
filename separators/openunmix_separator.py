import os
import tempfile
import logging
import torch
import librosa
import soundfile as sf
import numpy as np
import time
import music_tag
import threading # <--- Added for smooth UI animation handling
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

            hw_label = f"[{str(target_device).upper()}] "
            # Helper to push clean updates down to the UI layout
            def send_progress(pct: int, msg: str):
                if progress_callback:
                    try:
                        progress_callback(pct, f"{hw_label}OpenUnmix: {msg}")
                    except Exception:
                        pass

            # 1. Metadata Generation Phase
            send_progress(5, "Loading track profile and extracting metadata...")
            original_tags = get_audio_metadata(input_path)
            v_tags = prepare_stem_metadata(original_tags, "Vocals")
            i_tags = prepare_stem_metadata(original_tags, "Instrumental")
            
            model_to_load = os.path.abspath(model) if os.path.isdir(model) else model

            # 2. Prediction (Returns float32 NumPy arrays)
            try:
                send_progress(12, "Parsing audio data matrix into memory arrays...")
                audio_np, _ = librosa.load(input_path, sr=44100, mono=False)
                if audio_np.ndim == 1: 
                    audio_np = np.stack([audio_np, audio_np], axis=0)
                
                send_progress(20, f"Commencing neural cross-analysis on target device ({target_device})...")
                
                # Setup an async ticker to animate the bar during silent full-file execution
                stop_ticker = threading.Event()
                def smooth_ui_ticker():
                    current_fake_pct = 22
                    while not stop_ticker.is_set() and current_fake_pct < 80:
                        send_progress(current_fake_pct, "Isolating vocal frequencies and residuals...")
                        time.sleep(1.2)  # Crawl forward every 1.2 seconds
                        current_fake_pct += 1

                ticker_thread = threading.Thread(target=smooth_ui_ticker, daemon=True)
                ticker_thread.start()

                try:
                    # Execute heavy PyTorch matrix isolation math
                    v_raw, i_raw = self._predict_core(audio_np, model_to_load, target_device)
                finally:
                    # Assure ticker thread terminates under all execution branches
                    stop_ticker.set()
                    ticker_thread.join(timeout=1.0)

            except (RuntimeError, torch.cuda.OutOfMemoryError):
                logging.warning("OpenUnmix OOM detected. Falling back to multi-pass memory chunking mode...")
                v_raw, i_raw = self._separate_by_chunks(input_path, model_to_load, target_device, progress_callback)

            # 3. Export Stems Phase
            send_progress(82, "Mapping decoupled frequency boundaries to target configurations...")
            stems = [
                (v_raw, f"{song_name}_OpenUnmix_{model}_vocals.{fmt}", v_tags, vocals_folder),
                (i_raw, f"{song_name}_OpenUnmix_{model}_instrumental.{fmt}", i_tags, instr_folder)
            ]

            final_filenames = []
            for idx, (data, target_name, tag_dict, out_folder) in enumerate(stems):
                stem_type = "vocal component" if idx == 0 else "instrumental residue"
                send_progress(85 + (idx * 7), f"Mastering and formatting {stem_type} layer...")
                
                save_path = get_unique_filename(os.path.join(out_folder, target_name))
                
                # --- Audio Formatting Logic ---
                if fmt.lower() in ['wav', 'flac']:
                    out_data = data.T 
                    if sr != 44100:
                        out_data = librosa.resample(out_data.T, orig_sr=44100, target_sr=sr).T
                    if channels == "Mono" and out_data.ndim > 1:
                        out_data = out_data.mean(axis=1)

                    st = {"32-bit": "FLOAT", "24-bit": "PCM_24", "16-bit": "PCM_16"}.get(bit_depth, "PCM_16")
                    sf.write(save_path, out_data, sr, subtype=st)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, data.T, 44100, subtype='FLOAT')
                        from pydub import AudioSegment
                        audio = AudioSegment.from_file(tmp.name)
                        if channels == "Mono": audio = audio.set_channels(1)
                        if audio.frame_rate != sr: audio = audio.set_frame_rate(sr)
                        audio.export(save_path, format=fmt, bitrate=bitrate)
                    os.unlink(tmp.name)

                self._apply_tags(save_path, tag_dict, f"OpenUnmix {model}")
                final_filenames.append(os.path.basename(save_path))

            send_progress(100, "Completing structural file validation checks!")
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
        chunk_len = 30 * 44100  
        fade_len = int(0.2 * 44100) 
        
        v_final = np.zeros_like(audio_np)
        i_final = np.zeros_like(audio_np)

        fade_in = np.linspace(0, 1, fade_len).reshape(1, -1)
        fade_out = np.linspace(1, 0, fade_len).reshape(1, -1)

        start = 0
        while start < total_samples:
            end = min(start + chunk_len, total_samples)
            chunk = audio_np[:, start:end]

            # Linear progress map matching the internal 15% to 80% boundary footprint
            pct = 15 + int((start / total_samples) * 65)
            if progress_callback:
                try:
                    progress_callback(pct, f"OpenUnmix: Processing chunk loop ({int(start/fs)}s / {int(total_samples/fs)}s)...")
                except Exception:
                    pass

            v_chunk, i_chunk = self._predict_core(chunk, model_str, device)
            
            # --- Crossfade Blending ---
            if start == 0:
                v_final[:, start:end] = v_chunk
                i_final[:, start:end] = i_chunk
            else:
                v_final[:, start:start+fade_len] = (
                    v_final[:, start:start+fade_len] * fade_out + 
                    v_chunk[:, :fade_len] * fade_in
                )
                i_final[:, start:start+fade_len] = (
                    i_final[:, start:start+fade_len] * fade_out + 
                    i_chunk[:, :fade_len] * fade_in
                )
                v_final[:, start+fade_len:end] = v_chunk[:, fade_len:]
                i_final[:, start+fade_len:end] = i_chunk[:, fade_len:]

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
        except Exception as e: 
            logging.warning(f"Tagging failed: {e}")