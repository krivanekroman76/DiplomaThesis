import os
import whisper
import librosa
import math
import torch
import gc
import logging

class WhisperTranscription:
    def __init__(self):
        self.current_model_name = None
        self.model = None

    def _get_device(self, device_choice):
        if device_choice in ["Auto", "GPU"]:
            if torch.cuda.is_available(): return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def load_model(self, model_name: str, device_choice: str, progress_callback=None):
        target_device = self._get_device(device_choice)
        
        if model_name != self.current_model_name or self.model is None or self.model.device.type != target_device:
            if self.model is not None:
                del self.model
                self.model = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): torch.mps.empty_cache()

            import sys
            base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.abspath(".")
            whisper_path = os.path.join(base_dir, "pretrained_models", "whisper")
            os.makedirs(whisper_path, exist_ok=True)
            
            # --- DOWNLOAD WARNING ---
            if progress_callback:
                progress_callback(5, f"[{target_device.upper()}] Whisper: Downloading/Loading model (May take a moment)...")
            
            logging.debug(f"Whisper: Loading '{model_name}' on {target_device.upper()}...")
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

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription (Model: {model_name}):\n")
                f.write(" ".join(full_text_blocks) + "\n\nTimestamps:\n")
                for seg in all_segments:
                    f.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text'].strip()}\n")

            del audio
            gc.collect()
            return True, os.path.basename(output_path)
            
        except Exception as e:
            logging.error(f"Whisper Error: {e}")
            if progress_callback: progress_callback(0, f"Error: {str(e)}")
            return False, None