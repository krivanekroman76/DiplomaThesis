import os
import torch
import librosa
import math
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from .utils import resolve_torch_device, clear_memory_cache, save_transcription_to_file

class Wav2Vec2Transcription:
    def __init__(self):
        self.current_model_name = None 
        self.processor = None
        self.model = None

    def transcribe(self, audio_path: str, output_path: str, model_name: str, device_choice="Auto", progress_callback=None):
        # Determine initial device
        target_device = resolve_torch_device(device_choice, return_string=True)
        
        try:
            return self._run_inference(audio_path, output_path, model_name, target_device, progress_callback)
        except Exception as e:
            # OOM Fallback logic
            if "out of memory" in str(e).lower() and target_device != "cpu":
                logging.warning("Wav2Vec2 OOM on GPU. Falling back to CPU...")
                clear_memory_cache()
                return self._run_inference(audio_path, output_path, model_name, "cpu", progress_callback)
            else:
                logging.error(f"Wav2Vec2 Error: {e}")
                return False, None

    def _run_inference(self, audio_path, output_path, model_name, device, progress_callback):
        prefix = f"[{str(device).upper()}]"
        
        # 1. Load Model & Processor
        if model_name != self.current_model_name or self.model is None:
            if progress_callback: progress_callback(5, f"{prefix} Wav2Vec2: Loading weights...")
            self.processor = Wav2Vec2Processor.from_pretrained(model_name)            
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
            self.current_model_name = model_name
            # Explicitly assign these so Pylance "sees" them
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)

        # Type Guard for Pylance
        if self.processor is None or self.model is None:
            raise RuntimeError("Failed to initialize Wav2Vec2 components.")

        # 2. Prepare Audio
        audio, _ = librosa.load(audio_path, sr=16000)
        
        # Precision Settings
        chunk_len = 20  # seconds
        overlap = 2     # seconds buffer
        sr = 16000
        
        total_samples = len(audio)
        step = chunk_len * sr
        
        full_transcription = []
        all_segments = []

        # 3. Processing Loop with Overlap
        with torch.no_grad():
            for start_sample in range(0, total_samples, step):
                end_sample = min(start_sample + step + (overlap * sr), total_samples)
                chunk = audio[start_sample:end_sample]

                if progress_callback:
                    percent = (start_sample / total_samples) * 85
                    progress_callback(10 + percent, f"{prefix} Wav2Vec2: Processing...")

                # Pylance Fix: Access feature_extractor directly
                inputs = self.feature_extractor(
                    chunk, 
                    sampling_rate=sr, 
                    return_tensors="pt", 
                    padding=True
                )
                
                input_values = inputs.input_values.to(device)
                logits = self.model(input_values).logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                text = self.tokenizer.batch_decode(predicted_ids)[0].lower()
                
                if text.strip():
                    full_transcription.append(text)
                    all_segments.append({
                        "start": start_sample / sr,
                        "end": (start_sample + step) / sr,
                        "text": text
                    })

        # 4. Finalize
        if progress_callback: progress_callback(95, f"{prefix} Wav2Vec2: Saving...")
        save_transcription_to_file(output_path, model_name, full_transcription, all_segments)
        
        clear_memory_cache()
        if progress_callback: progress_callback(100, f"{prefix} Wav2Vec2: Complete!")
        return True, os.path.basename(output_path)