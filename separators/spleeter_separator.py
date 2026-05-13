import os
import sys
import logging
import tempfile
import platform
import subprocess
from typing import Optional, Dict, Any, Tuple
from pydub import AudioSegment
from spleeter.separator import Separator

from .utils import (
    setup_ffmpeg_environment, 
    get_unique_filename, 
    get_audio_metadata, 
    prepare_stem_metadata,
    finalize_metadata,
    resolve_tensorflow_device,
    ProgressInterceptor,
    LogStreamer
)

class SpleeterSeparator:
    def __init__(self):
        setup_ffmpeg_environment() 
        self.model = 'spleeter:2stems'
        logging.info("Spleeter Wrapper initialized (Full Safety & Logging Mode)")

    def get_subprocess_flags(self) -> int:
        return 0x08000000 if platform.system() == "Windows" else 0

    def _process_manual_chunks(self, input_path: str, temp_dir: str, prefix: str, progress_callback: Any) -> Tuple[Optional[AudioSegment], Optional[AudioSegment]]:
        """
        Safety Layer: Processes audio in 60s segments to prevent system RAM exhaustion (OOM).
        """
        audio = AudioSegment.from_file(input_path)
        chunk_length_ms = 60 * 1000 
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        final_vocals: Optional[AudioSegment] = None
        final_instr: Optional[AudioSegment] = None

        for idx, chunk in enumerate(chunks):
            if progress_callback: 
                progress_callback(30 + int(40 * (idx / len(chunks))), f"{prefix} Spleeter: RAM Safety - Segment {idx + 1}/{len(chunks)}...")
            
            c_path = os.path.join(temp_dir, f"chunk_{idx}.wav")
            chunk.export(c_path, format="wav")
            
            cmd = ['spleeter', 'separate', '-p', self.model, '-o', temp_dir, c_path]
            subprocess.run(cmd, capture_output=True, env=os.environ.copy(), creationflags=self.get_subprocess_flags())

            v_chunk_p = os.path.join(temp_dir, f"chunk_{idx}", "vocals.wav")
            i_chunk_p = os.path.join(temp_dir, f"chunk_{idx}", "accompaniment.wav")
            
            if os.path.exists(v_chunk_p):
                cv = AudioSegment.from_file(v_chunk_p)
                ci = AudioSegment.from_file(i_chunk_p)
                if final_vocals is None or final_instr is None:
                    final_vocals, final_instr = cv, ci
                else:
                    final_vocals = final_vocals.append(cv, crossfade=10)
                    final_instr = final_instr.append(ci, crossfade=10)
        
        return final_vocals, final_instr

    def separate(self, input_path: str, song_name: str, vocals_folder: str, instr_folder: str, 
                  channels: str = "Stereo", fmt: str = "wav", sr: int = 44100, bitrate: str = "128k", 
                  device_choice: str = "Auto", flac_compression: int = 5, progress_callback: Any = None):
        
        spleeter_logger = logging.getLogger("spleeter")
        if not spleeter_logger.handlers:
            spleeter_logger.addHandler(ProgressInterceptor(progress_callback, "[SPLEETER]"))

        # Using Optional tells Pylance that None is allowed initially
        v_audio: Optional[AudioSegment] = None
        i_audio: Optional[AudioSegment] = None

        try:
            if not os.path.exists(input_path): return False, None, None

            # 1. Metadata & Device
            original_tags = get_audio_metadata(input_path)
            v_tags = finalize_metadata(prepare_stem_metadata(original_tags, "Vocals"), "Vocals", "Spleeter")
            i_tags = finalize_metadata(prepare_stem_metadata(original_tags, "Instrumental"), "Instrumental", "Spleeter")
            resolved_device = resolve_tensorflow_device(device_choice)
            logging.info(f"[Spleeter] Separation is requesting device: {device_choice} -> Resolved to: {resolved_device}")
            prefix = f"[{str(resolved_device).upper()}]"

            if progress_callback: progress_callback(10, f"{prefix} Spleeter: Initializing...")

            with tempfile.TemporaryDirectory() as temp_dir:
                orig_stderr = sys.stderr
                sys.stderr = LogStreamer(spleeter_logger)
                
                try:
                    # PHASE A: Primary API Attempt
                    try:
                        separator = Separator(self.model)
                        separator.separate_to_file(input_path, temp_dir)
                        
                        s_dir = os.path.splitext(os.path.basename(input_path))[0]
                        v_p, i_p = os.path.join(temp_dir, s_dir, "vocals.wav"), os.path.join(temp_dir, s_dir, "accompaniment.wav")
                        if os.path.exists(v_p):
                            v_audio, i_audio = AudioSegment.from_file(v_p), AudioSegment.from_file(i_p)
                    
                    except Exception as api_err:
                        # PHASE B: CLI Fallback if API fails
                        logging.warning(f"Spleeter API failed ({api_err}), trying CLI fallback...")
                        cmd = ['spleeter', 'separate', '-p', self.model, '-o', temp_dir, input_path]
                        res = subprocess.run(cmd, capture_output=True, text=True, creationflags=self.get_subprocess_flags())
                        
                        if res.returncode != 0 and ("memory" in res.stderr.lower() or "oom" in res.stderr.lower()):
                            # PHASE C: Manual Chunking
                            # FIXED: Assigning to temp variables first to satisfy Pylance Type narrowing
                            res_v, res_i = self._process_manual_chunks(input_path, temp_dir, prefix, progress_callback)
                            v_audio, i_audio = res_v, res_i
                        else:
                            s_dir = os.path.splitext(os.path.basename(input_path))[0]
                            v_p = os.path.join(temp_dir, s_dir, "vocals.wav")
                            if os.path.exists(v_p):
                                v_audio, i_audio = AudioSegment.from_file(v_p), AudioSegment.from_file(os.path.join(temp_dir, s_dir, "accompaniment.wav"))
                finally:
                    sys.stderr = orig_stderr

                # 2. TYPE GUARD & POST-PROCESSING
                # This check ensures Pylance knows that from here on, audio is definitely NOT None
                if v_audio is None or i_audio is None:
                    return False, None, None

                # Re-assignment to localized names can help some Pylance versions clarify the type
                final_v: AudioSegment = v_audio
                final_i: AudioSegment = i_audio

                if channels == "Mono": 
                    final_v, final_i = final_v.set_channels(1), final_i.set_channels(1)
                
                if final_v.frame_rate != sr: 
                    final_v, final_i = final_v.set_frame_rate(sr), final_i.set_frame_rate(sr)

                # 3. EXPORT
                v_dest = get_unique_filename(os.path.join(vocals_folder, f"{song_name}_Spleeter_vocals.{fmt}"))
                i_dest = get_unique_filename(os.path.join(instr_folder, f"{song_name}_Spleeter_instrumental.{fmt}"))

                def export_with_meta(audio_seg: AudioSegment, path: str, tags: Optional[Dict[str, Any]]):
                    clean_tags: Dict[str, str] = {str(k): str(v[0]) if isinstance(v, list) else str(v) for k, v in (tags or {}).items()}
                    params = {"out_f": path, "format": fmt, "tags": clean_tags}
                    if fmt == "mp3": params["bitrate"] = bitrate
                    elif fmt == "flac": params["parameters"] = ["-compression_level", str(flac_compression)]
                    audio_seg.export(**params)

                export_with_meta(final_v, v_dest, v_tags)
                export_with_meta(final_i, i_dest, i_tags)

                if progress_callback: progress_callback(100, f"{prefix} Spleeter: Complete")
                return True, os.path.basename(v_dest), os.path.basename(i_dest)

        except Exception as e:
            logging.error(f"Spleeter error: {e}", exc_info=True)
            return False, None, None