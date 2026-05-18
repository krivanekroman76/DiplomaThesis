import os
import torch
import librosa
import logging
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from .utils import (
    resolve_torch_device, 
    clear_memory_cache, 
    save_transcription_to_file, 
    ProgressInterceptor,
    apply_vad # --- NEW IMPORT ---
)

class Wav2Vec2Transcription:
    def __init__(self):
        self.current_model_name = None 
        self.processor = None
        self.model = None

    def transcribe(self, audio_path: str, output_path: str, model_name: str, device_choice="Auto", progress_callback=None):
        
        # --- APPLY VAD STEP ---
        vad_output_path = audio_path.replace(".wav", "_vad.wav")
        if progress_callback:
            progress_callback(2, "Applying Voice Activity Detection...")
        
        processed_audio_path = apply_vad(audio_path, vad_output_path)
        # ----------------------

        target_device = resolve_torch_device(device_choice, return_string=True)
        
        try:
            return self._run_inference(processed_audio_path, output_path, model_name, target_device, progress_callback)
        except Exception as e:
            if "out of memory" in str(e).lower() and target_device != "cpu":
                logging.warning("Wav2Vec2 OOM on GPU. Falling back to CPU...")
                clear_memory_cache()
                return self._run_inference(processed_audio_path, output_path, model_name, "cpu", progress_callback)
            else:
                logging.error(f"Wav2Vec2 Error: {e}")
                return False, None
        finally:
            # --- CLEANUP TEMP VAD FILE ---
            if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                try:
                    os.remove(processed_audio_path)
                except Exception as cleanup_error:
                    logging.warning(f"Failed to delete temp VAD file: {cleanup_error}")

    def _run_inference(self, audio_path, output_path, model_name, device, progress_callback):
        prefix = f"[{str(device).upper()}]"
        
        # 1. Setup Progress Interceptor
        total_duration = librosa.get_duration(filename=audio_path)
        handler = ProgressInterceptor(progress_callback, device=device, tool_name="Wav2Vec2", total_duration=int(total_duration))
        logging.getLogger().addHandler(handler)

        try:
            # 2. Load Model & Processor (Progress 5-10%)
            if model_name != self.current_model_name or self.model is None:
                if progress_callback: progress_callback(5, f"{prefix} Wav2Vec2: Loading weights...")
                self.processor = Wav2Vec2Processor.from_pretrained(model_name)            
                self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
                self.current_model_name = model_name
                self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
                self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)

            if self.processor is None or self.model is None:
                raise RuntimeError("Failed to initialize Wav2Vec2 components.")

            # 3. Prepare Audio
            audio, _ = librosa.load(audio_path, sr=16000)
            chunk_len = 20  # seconds
            overlap = 2     # seconds buffer
            sr = 16000
            total_samples = len(audio)
            step = chunk_len * sr
            
            full_transcription = []
            all_segments = []

            # 4. Processing Loop (Progress 10-90%)
            with torch.no_grad():
                for start_sample in range(0, total_samples, step):
                    end_sample = min(start_sample + step + (overlap * sr), total_samples)
                    chunk = audio[start_sample:end_sample]

                    if progress_callback:
                        # Scale progress between 10% and 90%
                        percent = (start_sample / total_samples) * 80 
                        progress_callback(int(10 + percent), f"{prefix} Wav2Vec2: Processing {model_name}...")

                    inputs = self.feature_extractor(chunk, sampling_rate=sr, return_tensors="pt", padding=True)
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

            # 5. Finalize (Progress 95-100%)
            if progress_callback: progress_callback(95, f"{prefix} Wav2Vec2: Saving to file...")
            save_transcription_to_file(output_path, model_name, full_transcription, all_segments)
            
            clear_memory_cache()
            if progress_callback: progress_callback(100, f"{prefix} Wav2Vec2: Complete!")
            return True, os.path.basename(output_path)

        finally:
            logging.getLogger().removeHandler(handler)