import os
import sys
import re
import logging
import torch
import tempfile
import pathlib
import time
import soundfile as sf
import librosa
import music_tag
from typing import Optional, Tuple
from pydub import AudioSegment
from demucs.separate import main as demucs_main
import demucs.pretrained

from .utils import (
    setup_ffmpeg_environment, 
    get_unique_filename, 
    resolve_torch_device, 
    get_audio_metadata, 
    prepare_stem_metadata,
    clear_memory_cache,
    ProgressInterceptor
)

class TqdmInterceptor:
    """
    Hijacks the raw sys.stderr stream to snatch percentages from tqdm progress bars 
    that bypass standard Python logging.
    """
    def __init__(self, callback, original_stream, device: str):
        self.callback = callback
        self.original_stream = original_stream
        self.buffer = ""
        # Regex to find numbers immediately followed by a % sign (e.g., " 45%|")
        self.pct_regex = re.compile(r'(\d{1,3})%')
        # Format device string nicely (e.g., cuda -> CUDA, cpu -> CPU)
        self.hw_label = f"[{str(device).upper()}] "

    def write(self, buf):
        # 1. Still print to the actual console so your terminal looks normal
        self.original_stream.write(buf) 
        
        # 2. Add text to our snatcher buffer
        self.buffer += buf
        
        # 3. tqdm uses \r to refresh the line. When we see one, parse the buffer!
        if '\r' in buf or '\n' in buf:
            match = self.pct_regex.search(self.buffer)
            if match and self.callback:
                pct = int(match.group(1))
                scaled_pct = 10 + int(pct * 0.8)
                try:
                    # Added hw_label here
                    self.callback(scaled_pct, f"{self.hw_label}Demucs: Processing audio chunks... {pct}%")
                except Exception:
                    pass
            self.buffer = ""

    def flush(self):
        self.original_stream.flush()


class DemucsSeparator:
    def __init__(self):
        setup_ffmpeg_environment()
        logging.getLogger("torch").setLevel(logging.ERROR)
        logging.info("Demucs 4.0.1 (Hybrid Transformer) initialized.")

    def separate(self, input_path: str, song_name: str, vocals_folder: str, instr_folder: str, 
                 model="htdemucs", channels="Stereo", fmt="wav", sr=44100, 
                 bitrate="128k", bit_depth="16-bit", shifts=1, 
                 overlap=0.1, device_choice="AUTO", flac_compression=5,
                 progress_callback=None, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        
        target_device = resolve_torch_device(device_choice.upper(), return_string=True)
        user_segment = kwargs.get('segment')

        # --- FUTURE-PROOF METADATA QUERY ---
        try:
            model_obj = demucs.pretrained.get_model(model)
            native_segment = getattr(model_obj, 'segment', 7.8) 
            logging.info(f"[Demucs] Model native segment limit: {native_segment}")
        except Exception as e:
            native_segment = 7.8 if "htdemucs" in model.lower() else 44.0
            logging.warning(f"[Demucs] Could not query model metadata ({e}). Using fallback limit: {native_segment}")

        # --- ADAPTIVE SEGMENT LOGIC ---
        if user_segment is None or user_segment > native_segment:
            current_segment = int(native_segment) 
        else:
            current_segment = int(user_segment)

        attempts = [
            {"device": target_device, "seg": current_segment, "label": f"{target_device}"},
            {"device": "cpu", "seg": current_segment, "label": "CPU Fallback"},
            {"device": "cpu", "seg": min(current_segment, 4), "label": "CPU Safe-Mode"}
        ]

        last_exception = None
        for attempt in attempts:
            if attempt["label"] == "CPU Fallback" and target_device == "cpu":
                continue

            try:
                logging.info(f"[Demucs] Attempting {attempt['label']} | Segment: {attempt['seg']}")
                if progress_callback:
                    progress_callback(5, f"Demucs: Starting {attempt['label']}...")

                return self._execute_core(
                    input_path, song_name, vocals_folder, instr_folder,
                    model, channels, fmt, sr, bitrate, bit_depth, shifts,
                    overlap, attempt["device"], attempt["seg"], flac_compression, progress_callback
                )

            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()
                
                if "longer segment than it was trained for" in err_msg:
                    logging.error("Segment length mismatch detected. Retrying with architecture-safe segment.")
                    for a in attempts: a["seg"] = 7.0 
                    continue

                if any(x in err_msg for x in ["out of memory", "alloc", "memory limit", "reallocate"]):
                    logging.warning(f"Memory exhaustion during {attempt['label']}. Trying next fallback...")
                    clear_memory_cache()
                    time.sleep(1)
                    continue
                else:
                    logging.error(f"Non-memory error in Demucs: {e}")
                    break

        logging.error(f"Demucs exhausted all fallback strategies. Final error: {last_exception}")
        return False, None, None

    def _execute_core(self, input_path, song_name, vocals_folder, instr_folder, 
                      model, channels, fmt, sr, bitrate, bit_depth, shifts, 
                      overlap, device, segment, flac_compression, progress_callback):
        
        root_logger = logging.getLogger()
        handler = ProgressInterceptor(progress_callback, device=device, tool_name="Demucs")
        root_logger.addHandler(handler)
        
        with tempfile.TemporaryDirectory(prefix="demucs_v4_") as temp_dir:
            try:
                original_tags = get_audio_metadata(input_path)
                v_tags = prepare_stem_metadata(original_tags, "Vocals")
                i_tags = prepare_stem_metadata(original_tags, "Instrumental")

                demucs_args = [
                    "-n", model, 
                    "-o", temp_dir, 
                    input_path,
                    "--shifts", str(shifts), 
                    "--overlap", str(overlap),
                    "--two-stems", "vocals", 
                    "-d", device,
                    "--segment", str(int(float(segment))),
                    "--float32"
                ]

                # --- NEW INTERCEPTION LOGIC ---
                # Backup the real console stream
                original_stderr = sys.stderr 
                # Replace it with our snatcher class
                sys.stderr = TqdmInterceptor(progress_callback, original_stderr, device) 
                
                try:
                    # Run Demucs while our snatcher is active!
                    demucs_main(demucs_args)
                finally:
                    # VERY IMPORTANT: Always give stderr back to the OS!
                    sys.stderr = original_stderr
                # ------------------------------

                input_stem = pathlib.Path(input_path).stem
                sep_folder = os.path.join(temp_dir, model, input_stem)
                v_src = os.path.join(sep_folder, "vocals.wav")
                i_src = os.path.join(sep_folder, "no_vocals.wav")
                
                if not os.path.exists(i_src): 
                    i_src = os.path.join(sep_folder, "instrumental.wav")

                if not os.path.exists(v_src) or not os.path.exists(i_src):
                    raise RuntimeError("Demucs execution finished but files are missing.")

                if progress_callback:
                    progress_callback(90, "Exporting and resampling audio stems...")

                stems = [
                    (v_src, f"{song_name}_Demucs_{model}_vocals.{fmt}", v_tags, vocals_folder),
                    (i_src, f"{song_name}_Demucs_{model}_instrumental.{fmt}", i_tags, instr_folder)
                ]

                final_filenames = []
                for src, target_name, tag_dict, out_folder in stems:
                    save_path = get_unique_filename(os.path.join(out_folder, target_name))
                    
                    if fmt.lower() in ['wav', 'flac']:
                        data, native_sr = sf.read(src, dtype='float32')
                        if native_sr != sr:
                            data = librosa.resample(data.T, orig_sr=native_sr, target_sr=sr).T
                        if channels == "Mono" and data.ndim > 1:
                            data = data.mean(axis=1)

                        subtype = {"32-bit": "FLOAT", "24-bit": "PCM_24", "16-bit": "PCM_16"}.get(bit_depth, "PCM_16")
                        sf.write(save_path, data, sr, subtype=subtype)
                    else:
                        audio = AudioSegment.from_file(src)
                        if channels == "Mono": audio = audio.set_channels(1)
                        if audio.frame_rate != sr: audio = audio.set_frame_rate(sr)
                        audio.export(save_path, format=fmt, bitrate=bitrate)

                    self._apply_tags(save_path, tag_dict, model)
                    final_filenames.append(os.path.basename(save_path))

                if progress_callback:
                    progress_callback(100, "Finalizing track validation!")

                return True, final_filenames[0], final_filenames[1]

            finally:
                root_logger.removeHandler(handler)
                clear_memory_cache()

    def _apply_tags(self, file_path: str, tags: dict, model: str):
        try:
            f = music_tag.load_file(file_path)
            if f:
                f['title'] = tags.get('title', 'Unknown')
                f['artist'] = tags.get('artist', 'Unknown')
                f['album'] = tags.get('album', 'Separated Stems')
                f['comment'] = f"Separated by Demucs {model}"
                if tags.get('year'): f['year'] = tags['year']
                if tags.get('genre'): f['genre'] = tags['genre']
                f.save()
        except Exception as e:
            logging.warning(f"Metadata Tagging failed for {file_path}: {e}")