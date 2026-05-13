import os
import gc
import sys
import shutil
import logging
import torch
from typing import Optional, Tuple
from pydub import AudioSegment
from demucs.separate import main as demucs_main

from .utils import (
    setup_ffmpeg_environment, 
    get_unique_filename, 
    resolve_torch_device, 
    get_audio_metadata, 
    prepare_stem_metadata,
    finalize_metadata,
    clear_memory_cache,
    ProgressInterceptor,
    LogStreamer
)

class DemucsSeparator:
    def __init__(self):
        setup_ffmpeg_environment()
        logging.info("Demucs API Wrapper initialized (With OOM Fallback)")

    def separate(self, input_path: str, song_name: str, vocals_folder: str, instr_folder: str, 
                 model="htdemucs", channels="Stereo", fmt="wav", sr=44100, 
                 bitrate="128k", bit_depth="16-bit", shifts=1, 
                 overlap=0.1, device_choice="Auto", flac_compression=5,
                 progress_callback=None, **kwargs) -> Tuple[bool, Optional[str], Optional[str]]:
        
        # Resolve initial device
        target_device = resolve_torch_device(device_choice, return_string=True)
        
        try:
            # Attempt separation
            return self._run_separation_logic(
                input_path, song_name, vocals_folder, instr_folder,
                model, channels, fmt, sr, bitrate, bit_depth, shifts,
                overlap, target_device, flac_compression, progress_callback
            )
        except Exception as e:
            error_msg = str(e).lower()
            # Check specifically for CUDA/MPS Out of Memory
            if ("out of memory" in error_msg or "alloc" in error_msg) and target_device != "cpu":
                logging.warning(f"Demucs OOM on {target_device}. Falling back to CPU...")
                
                if progress_callback:
                    progress_callback(15, f"[CPU FALLBACK] VRAM Full. Restarting on CPU (Slow)...")
                
                # Full system flush before retry
                clear_memory_cache()
                
                # Retry specifically on CPU
                return self._run_separation_logic(
                    input_path, song_name, vocals_folder, instr_folder,
                    model, channels, fmt, sr, bitrate, bit_depth, shifts,
                    overlap, "cpu", flac_compression, progress_callback
                )
            else:
                logging.error(f"Demucs Critical Error: {e}", exc_info=True)
                return False, None, None

    def _run_separation_logic(self, input_path, song_name, vocals_folder, instr_folder, 
                             model, channels, fmt, sr, bitrate, bit_depth, shifts, 
                             overlap, device, flac_compression, progress_callback):
        
        base_temp_out = os.path.join(os.getcwd(), f"temp_demucs_{song_name}")
        prefix = f"[{str(device).upper()}]"
        
        # Setup Interceptor
        demucs_logger = logging.getLogger("demucs")
        handler = ProgressInterceptor(progress_callback, prefix)
        demucs_logger.addHandler(handler)

        try:
            if not os.path.exists(input_path): return False, None, None

            # 1. Metadata Preparation
            original_tags = get_audio_metadata(input_path)
            v_tags = finalize_metadata(prepare_stem_metadata(original_tags, "Vocals"), "Vocals", "Demucs")
            i_tags = finalize_metadata(prepare_stem_metadata(original_tags, "Instrumental"), "Instrumental", "Demucs")

            # 2. Argument Construction
            demucs_args = [
                "-n", model, "-o", base_temp_out, input_path,
                "--shifts", str(shifts), "--overlap", str(overlap),
                "--two-stems", "vocals", "-d", device, "--clip-mode", "rescale"
            ]
            
            # Format Logic
            if fmt == "mp3":
                demucs_args.extend(["--mp3", "--mp3-bitrate", bitrate.replace("k", "")])
            elif fmt == "flac":
                demucs_args.append("--flac")
            
            if bit_depth == "24-bit": demucs_args.append("--int24")
            elif bit_depth == "32-bit": demucs_args.append("--float32")
            
            if progress_callback: progress_callback(10, f"{prefix} Demucs: Initializing AI Model...")
            
            # 3. Execute Demucs Core
            original_stderr = sys.stderr
            sys.stderr = LogStreamer(demucs_logger)
            try:
                demucs_main(demucs_args)
            finally:
                sys.stderr = original_stderr

            # 4. Finalizing & Export
            input_base = os.path.splitext(os.path.basename(input_path))[0]
            sep_folder = os.path.join(base_temp_out, model, input_base)
            
            # Demucs uses 'no_vocals' for the instrumental stem in --two-stems mode
            stems = [
                (os.path.join(sep_folder, f"vocals.{fmt}"), f"{song_name}_Demucs_vocals.{fmt}", v_tags, vocals_folder),
                (os.path.join(sep_folder, f"no_vocals.{fmt}"), f"{song_name}_Demucs_instrumental.{fmt}", i_tags, instr_folder)
            ]

            final_paths = []
            for src, filename, tags, folder in stems:
                if not os.path.exists(src):
                    # Fallback check: some models use 'instrumental' instead of 'no_vocals'
                    alt_src = src.replace("no_vocals", "instrumental")
                    if os.path.exists(alt_src): src = alt_src

                seg = AudioSegment.from_file(src)
                if channels == "Mono": seg = seg.set_channels(1)
                if seg.frame_rate != sr: seg = seg.set_frame_rate(sr)
                
                dest = get_unique_filename(os.path.join(folder, filename))
                
                export_params = {"out_f": dest, "format": fmt, "tags": tags}
                if fmt == "flac": export_params["parameters"] = ["-compression_level", str(flac_compression)]
                
                seg.export(**export_params)
                final_paths.append(os.path.basename(dest))

            return True, final_paths[0], final_paths[1]

        finally:
            # Cleanup
            demucs_logger.removeHandler(handler)
            if os.path.exists(base_temp_out): 
                shutil.rmtree(base_temp_out, ignore_errors=True)
            clear_memory_cache()