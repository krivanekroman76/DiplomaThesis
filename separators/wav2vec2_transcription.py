import os
import logging
import torch
import librosa
import math
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from .utils import resolve_torch_device, clear_memory_cache, save_transcription_to_file

class Wav2Vec2Transcription:
    def __init__(self):
        self.current_model_name = None 
        self.processor = None
        self.model = None

    def transcribe(self, audio_path: str, output_path: str, model_name: str, device_choice="Auto", progress_callback=None):
        try:
            target_device = resolve_torch_device(device_choice, return_string=False)
            prefix = f"[{target_device.type.upper()}]"

            if model_name != self.current_model_name or self.model is None or self.model.device.type != target_device.type:
                if progress_callback:
                    progress_callback(5, f"{prefix} Wav2Vec2: Downloading/Loading weights...")
                
                logging.info(f"Wav2Vec2: Loading '{model_name}' on {target_device.type.upper()}...")
                if self.model is not None:
                    del self.model
                    del self.processor
                    clear_memory_cache()

                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(target_device)
                self.current_model_name = model_name

            if not os.path.exists(audio_path):
                return False, None

            audio, sr = librosa.load(audio_path, sr=16000)
            chunk_length_s = 10
            chunk_size = chunk_length_s * 16000
            total_samples = len(audio)
            total_chunks = math.ceil(total_samples / chunk_size)

            full_transcription = []
            all_segments = []

            with torch.no_grad():
                for chunk_index, i in enumerate(range(0, total_samples, chunk_size)):
                    if progress_callback:
                        percent_done = (chunk_index / total_chunks) * 80
                        progress_callback(10 + percent_done, f"{prefix} Wav2Vec2: Transcribing chunk {chunk_index + 1} of {total_chunks}...")

                    audio_chunk = audio[i : i + chunk_size]
                    inputs = self.processor(audio_chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                    logits = self.model(inputs.input_values.to(target_device)).logits
                    
                    predicted_ids = torch.argmax(logits, dim=-1)
                    chunk_text = self.processor.batch_decode(predicted_ids)[0]
                    
                    if chunk_text:
                        text_lower = chunk_text.lower()
                        full_transcription.append(text_lower)
                        start_t = chunk_index * chunk_length_s
                        all_segments.append({"start": start_t, "end": start_t + chunk_length_s, "text": text_lower})

            if progress_callback: progress_callback(95, f"{prefix} Wav2Vec2: Saving transcription...")

            save_transcription_to_file(output_path, model_name, full_transcription, all_segments)

            del audio
            del inputs
            del logits
            clear_memory_cache()
            
            if progress_callback: progress_callback(100, f"{prefix} Wav2Vec2: Complete!")
            logging.info(f"Wav2Vec2: Transcription saved to {output_path}")
            return True, os.path.basename(output_path)
            
        except Exception as e:
            logging.error(f"Wav2Vec2 Error: {e}", exc_info=True)
            if progress_callback: progress_callback(0, f"Error: {str(e)}")
            return False, None