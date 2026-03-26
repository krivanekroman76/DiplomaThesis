import os
import logging
import torch
import librosa
import gc # Import the garbage collector
import math
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class Wav2Vec2Transcription:
    def __init__(self, default_model="facebook/wav2vec2-base-960h"):
        # Store current model state
        self.current_model_name = None 
        self.processor = None
        self.model = None
        
        # Apple Silicon / CPU detection
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initial load
        self._load_model_weights(default_model)

    def _load_model_weights(self, model_name):
        """Internal helper to swap weights in memory."""
        if model_name == self.current_model_name:
            return

        print(f"[INFO] Wav2Vec2: Switching model to '{model_name}'...")
        try:
            # Clear old model from memory/cache before loading new one
            if self.model is not None:
                del self.model
                del self.processor
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            self.processor = Wav2Vec2Processor.from_pretrained(model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
            self.current_model_name = model_name
            print(f"[INFO] Wav2Vec2: Successfully loaded on {self.device}.")
        except Exception as e:
            print(f"[ERROR] Wav2Vec2: Failed to load {model_name}: {e}")

    def _get_device(self, device_choice):
        if device_choice in ["Auto", "GPU"]:
            if torch.cuda.is_available(): return torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def transcribe(self, audio_path: str, output_path: str, model_name: str, device_choice="Auto", progress_callback=None):
        try:
            target_device = self._get_device(device_choice)
            prefix = f"[{target_device.type.upper()}]"

            if model_name != self.current_model_name or self.model is None or self.model.device.type != target_device.type:
                if progress_callback:
                    progress_callback(5, f"{prefix} Wav2Vec2: Downloading/Loading weights...")
                
                logging.debug(f"Wav2Vec2: Loading on {target_device.type.upper()}")
                if self.model is not None:
                    del self.model
                    del self.processor
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

                self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(target_device)
                self.current_model_name = model_name

            if not os.path.exists(audio_path):
                return False, None # <-- Changed to return tuple

            # 1. Load the audio
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 2. Define chunk parameters
            chunk_length_s = 10
            chunk_size = chunk_length_s * 16000
            total_samples = len(audio)
            
            full_transcription = []

            # Calculate total chunks for our progress bar
            total_chunks = math.ceil(total_samples / chunk_size)

            # 3. Process in chunks
            with torch.no_grad():
                for chunk_index, i in enumerate(range(0, total_samples, chunk_size)):
                    
                    # --- PROGRESS BAR UPDATE ---
                    if progress_callback:
                        percent_done = (chunk_index / total_chunks) * 80
                        current_progress = 10 + percent_done
                        
                        progress_callback(
                            current_progress, 
                            f"Wav2Vec2: Transcribing chunk {chunk_index + 1} of {total_chunks}..."
                        )
                    # ---------------------------

                    audio_chunk = audio[i : i + chunk_size]
                    
                    inputs = self.processor(audio_chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                    logits = self.model(inputs.input_values.to(self.device)).logits
                    
                    predicted_ids = torch.argmax(logits, dim=-1)
                    chunk_text = self.processor.batch_decode(predicted_ids)[0]
                    
                    if chunk_text:
                        full_transcription.append(chunk_text.lower())

            if progress_callback:
                progress_callback(95, "Wav2Vec2: Saving transcription...")

            # 4. Save to file (Matching Whisper Format)
            with open(output_path, "w", encoding="utf-8") as f:
                # Write the full block of text first
                f.write(f"Transcription (Model: {model_name}):\n")
                f.write(" ".join(full_transcription) + "\n\n")
                
                # Write the generated chunk timestamps
                f.write("Timestamps:\n")
                for chunk_index, text in enumerate(full_transcription):
                    start_t = chunk_index * chunk_length_s
                    end_t = start_t + chunk_length_s
                    f.write(f"{start_t:.2f}s - {end_t:.2f}s: {text}\n")
                
            # 5. Cleanup RAM
            del audio
            del inputs
            del logits
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            if progress_callback:
                progress_callback(100, "Wav2Vec2: Transcription Complete!")
                
            # Extract just the filename from the full path
            filename = os.path.basename(output_path)
            return True, filename # <-- Return success and filename
            
        except Exception as e:
            print(f"[ERROR] Wav2Vec2 transcription failed: {e}")
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")
            return False, None # <-- Changed to return tuple