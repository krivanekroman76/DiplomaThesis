import os
import sys
import whisper
import librosa
import math
import logging
import torch
import re
from .utils import resolve_torch_device, clear_memory_cache, save_transcription_to_file

class WhisperTranscription:
    def __init__(self, custom_models_dir=None):
        self.current_model_name = None
        self.model = None
        self.models_dir = custom_models_dir or os.path.join(os.getcwd(), "Models")

    def transcribe(self, audio_path, output_path, model_name, language="auto", device_choice="Auto", progress_callback=None):
        target_device = resolve_torch_device(device_choice, return_string=True)
        
        try:
            return self._run_inference(audio_path, output_path, model_name, language, target_device, progress_callback)
        except Exception as e:
            if ("out of memory" in str(e).lower()) and target_device != "cpu":
                logging.warning(f"Whisper OOM on {target_device}. Falling back to CPU...")
                if progress_callback:
                    progress_callback(10, f"[CPU FALLBACK] VRAM Full. Switching to CPU...")
                clear_memory_cache()
                return self._run_inference(audio_path, output_path, model_name, language, "cpu", progress_callback)
            else:
                logging.error(f"Whisper Error: {e}")
                return False, None

    def _run_inference(self, audio_path, output_path, model_name, language, device, progress_callback):
        prefix = f"[{str(device).upper()}]"
        
        # Load Model
        if self.current_model_name != model_name or self.model is None:
            if progress_callback: progress_callback(5, f"{prefix} Whisper: Loading {model_name}...")
            whisper_path = os.path.join(self.models_dir, "whisper")
            self.model = whisper.load_model(model_name, device=device, download_root=whisper_path)
            self.current_model_name = model_name

        # Prepare Audio
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # French Rap Optimization: Provide a prompt to guide the AI on slang/style
        prompt = "Transcription de musique, rap français, paroles précises." if language in ["fr", "auto"] else ""
        
        # Transcription Logic
        lang_param = None if language.lower() == "auto" else language
        
        # We use Whisper's internal long-form transcription logic (better than manual chunking)
        # but we use a ProgressHook to intercept status if needed.
        if progress_callback: progress_callback(20, f"{prefix} Whisper: Processing Audio...")
        
        result = self.model.transcribe(
            audio_path,
            language=lang_param,
            initial_prompt=prompt,
            temperature=0.0,
            condition_on_previous_text=True,
            verbose=False # Keep console clean
        )

        if progress_callback: progress_callback(90, f"{prefix} Whisper: Formatting Output...")
        
        full_text = [result["text"]]
        segments = result["segments"] # Contains start, end, text
        
        save_transcription_to_file(output_path, model_name, full_text, segments)
        
        clear_memory_cache()
        if progress_callback: progress_callback(100, f"{prefix} Whisper: Complete!")
        return True, os.path.basename(output_path)