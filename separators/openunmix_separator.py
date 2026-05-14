import os
import tempfile
import logging
import torch
import librosa
import soundfile as sf
import numpy as np
from typing import Optional, List
from pydub import AudioSegment
from openunmix import predict

from .utils import (
    setup_ffmpeg_environment, 
    get_unique_filename, 
    resolve_torch_device,
    get_audio_metadata, 
    prepare_stem_metadata,
    finalize_metadata,
    clear_memory_cache
)

class OpenUnmixSeparator:
    def __init__(self):
        setup_ffmpeg_environment()
        logging.info("OpenUnmix initialized (RAM-Safety Chunking Mode)")

    def separate(self, input_path: str, song_name: str, vocals_folder: str, instr_folder: str, 
                 model="umx", channels="Stereo", fmt="wav", sr=44100, bitrate="128k", 
                 device_choice="Auto", flac_compression=5, progress_callback=None, **kwargs):  
        try:
            if not os.path.exists(input_path): return False, None, None

            # 1. Initialization
            original_tags = get_audio_metadata(input_path)
            v_tags = finalize_metadata(prepare_stem_metadata(original_tags, "Vocals"), "Vocals", "OpenUnmix")
            i_tags = finalize_metadata(prepare_stem_metadata(original_tags, "Instrumental"), "Instrumental", "OpenUnmix")

            target_device = resolve_torch_device(device_choice, return_string=True)
            prefix = f"[{str(target_device).upper()}]"
            
            if progress_callback: progress_callback(5, f"{prefix} OpenUnmix: Loading Audio...")

            # Load audio with pydub for easy chunking
            full_audio = AudioSegment.from_file(input_path)
            duration_ms = len(full_audio)
            chunk_length_ms = 30 * 1000  # 30-second chunks for high safety
            
            # Calculate chunks
            chunks = [full_audio[i:i + chunk_length_ms] for i in range(0, duration_ms, chunk_length_ms)]
            num_chunks = len(chunks)
            
            v_final: Optional[AudioSegment] = None
            i_final: Optional[AudioSegment] = None

            model_to_load = os.path.abspath(model) if os.path.isdir(model) else model

            with tempfile.TemporaryDirectory() as temp_dir:
                for idx, chunk in enumerate(chunks):
                    # Update progress dynamically based on chunk index
                    # Range: 10% to 85%
                    percent = 10 + int((idx / num_chunks) * 75)
                    if progress_callback:
                        # Manual formatting to match the interceptor's style
                        display_text = f"{prefix} OpenUnmix Separating: {percent}%"
                        progress_callback(percent, display_text)

                    # Export chunk to temp wav for librosa/umx
                    chunk_path = os.path.join(temp_dir, f"c_{idx}.wav")
                    chunk.export(chunk_path, format="wav")
                    
                    # Load into numpy for UMX
                    audio_np, _ = librosa.load(chunk_path, sr=44100, mono=False)
                    if audio_np.ndim == 1: 
                        audio_np = np.stack([audio_np, audio_np], axis=0)

                    # Neural Prediction
                    estimates = predict.separate(
                        audio=torch.as_tensor(audio_np).float(),
                        rate=44100,
                        model_str_or_path=model_to_load,
                        targets=['vocals'], 
                        residual=True, 
                        device=target_device 
                    )

                    # Convert back to AudioSegments
                    v_raw = estimates['vocals'].detach().cpu().numpy().squeeze()
                    i_raw = estimates['residual'].detach().cpu().numpy().squeeze()
                    
                    v_tmp_p = os.path.join(temp_dir, f"v_{idx}.wav")
                    i_tmp_p = os.path.join(temp_dir, f"i_{idx}.wav")
                    sf.write(v_tmp_p, v_raw.T, 44100)
                    sf.write(i_tmp_p, i_raw.T, 44100)
                    
                    v_seg_chunk = AudioSegment.from_wav(v_tmp_p)
                    i_seg_chunk = AudioSegment.from_wav(i_tmp_p)

                    # Merge with final audio using a small crossfade to prevent pops
                    if v_final is None or i_final is None:
                        v_final, i_final = v_seg_chunk, i_seg_chunk
                    else:
                        v_final = v_final.append(v_seg_chunk, crossfade=200)
                        i_final = i_final.append(i_seg_chunk, crossfade=200) 

                    # Periodic cache clearing within the loop
                    clear_memory_cache()

                # 2. Post-Processing & Export
                if v_final is None or i_final is None: return False, None, None
                
                if progress_callback: progress_callback(90, f"{prefix} OpenUnmix: Exporting Stems...")

                if channels == "Mono":
                    v_final, i_final = v_final.set_channels(1), i_final.set_channels(1)
                if v_final.frame_rate != sr:
                    v_final, i_final = v_final.set_frame_rate(sr), i_final.set_frame_rate(sr)

                v_dest = get_unique_filename(os.path.join(vocals_folder, f"{song_name}_OpenUnmix_{model}_vocals.{fmt}"))
                i_dest = get_unique_filename(os.path.join(instr_folder, f"{song_name}_OpenUnmix_{model}_instrumental.{fmt}"))

                # Reusable export params
                def final_export(seg, path, tags):
                    params = {"out_f": path, "format": fmt, "tags": tags}
                    if fmt == "mp3": params["bitrate"] = bitrate
                    elif fmt == "flac": params["parameters"] = ["-compression_level", str(flac_compression)]
                    seg.export(**params)

                final_export(v_final, v_dest, v_tags)
                final_export(i_final, i_dest, i_tags)

                if progress_callback: progress_callback(100, f"{prefix} OpenUnmix: Success!")
                return True, os.path.basename(v_dest), os.path.basename(i_dest)

        except Exception as e:
            logging.error(f"OpenUnmix Error: {e}", exc_info=True)
            return False, None, None
        finally:
            clear_memory_cache()