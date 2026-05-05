import os
import sys
import whisper
import librosa
import math
import logging
from .utils import resolve_torch_device, clear_memory_cache, save_transcription_to_file

class WhisperTranscription:
    # Accept custom_models_dir just like Vosk!
    def __init__(self, custom_models_dir=None):
        self.current_model_name = None
        self.model = None
        
        # Mimic your Vosk logic to find the base directory
        if getattr(sys, 'frozen', False):
            base_project_dir = os.path.dirname(sys.executable)
        else:
            base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
        self.models_dir = custom_models_dir if custom_models_dir else os.path.join(base_project_dir, "Models")

    def load_model(self, model_name: str, device_choice: str, progress_callback=None):
        target_device = resolve_torch_device(device_choice, return_string=True)
        
        if model_name != self.current_model_name or self.model is None or self.model.device.type != target_device:
            if self.model is not None:
                del self.model
                self.model = None
                clear_memory_cache()

            # --- THE FIX: Point exactly to the Models/whisper folder ---
            whisper_path = os.path.join(self.models_dir, "whisper")
            os.makedirs(whisper_path, exist_ok=True)
            
            if progress_callback:
                progress_callback(5, f"[{target_device.upper()}] Whisper: Downloading/Loading model...")
            
            logging.info(f"Whisper: Loading '{model_name}' on {target_device.upper()} from {whisper_path}...")
            # Now download_root perfectly matches your download script
            self.model = whisper.load_model(model_name, device=target_device, download_root=whisper_path)
            self.current_model_name = model_name

        return target_device

    def transcribe(self, audio_path, output_path, model_name, language="auto", device_choice="Auto", progress_callback=None):
        try:
            target_device = self.load_model(model_name, device_choice, progress_callback)
            prefix = f"[{target_device.upper()}]"
            
            if progress_callback:
                progress_callback(15, f"{prefix} Whisper: Preparing audio chunks...")
            audio, sr = librosa.load(audio_path, sr=16000)
            
            chunk_length_s = 30
            chunk_samples = chunk_length_s * 16000
            total_samples = len(audio)
            total_chunks = math.ceil(total_samples / chunk_samples)

            full_text_blocks = []
            all_segments = []
            previous_context = ""
            lang_param = None if language == "auto" else language

            for chunk_idx, i in enumerate(range(0, total_samples, chunk_samples)):
                if progress_callback:
                    percent_done = (chunk_idx / total_chunks) * 80
                    progress_callback(15 + percent_done, f"{prefix} Whisper: Transcribing chunk {chunk_idx + 1}/{total_chunks}...")

                chunk = audio[i : i + chunk_samples]
                result = self.model.transcribe(chunk, language=lang_param, initial_prompt=previous_context)

                text = result["text"].strip()
                if text:
                    full_text_blocks.append(text)
                    previous_context = text[-200:] 

                time_offset = chunk_idx * chunk_length_s
                for seg in result["segments"]:
                    seg["start"] += time_offset
                    seg["end"] += time_offset
                    all_segments.append(seg)

            if progress_callback: progress_callback(95, f"{prefix} Whisper: Formatting file...")

            save_transcription_to_file(output_path, model_name, full_text_blocks, all_segments)

            del audio
            clear_memory_cache()
            logging.info(f"Whisper: Transcription saved to {output_path}")
            return True, os.path.basename(output_path)
            
        except Exception as e:
            logging.error(f"Whisper Error: {e}", exc_info=True)
            if progress_callback: progress_callback(0, f"Error: {str(e)}")
            return False, None