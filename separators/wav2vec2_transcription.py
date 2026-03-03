import os
import torch
import librosa
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

    def transcribe(self, audio_path: str, output_path: str, model_name: str):
        """
        Transcribes audio, switching models if necessary.
        :param model_name: The technical string from your GUI settings (e.g., 'facebook/wav2vec2-large-960h').
        """
        try:
            # Check if we need to swap weights
            if model_name and model_name != self.current_model_name:
                self._load_model_weights(model_name)

            if not os.path.exists(audio_path):
                return False

            # Processing logic
            audio, _ = librosa.load(audio_path, sr=16000)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            
            with torch.no_grad():
                logits = self.model(inputs.input_values.to(self.device)).logits
            
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcription.lower())
                
            return True
        except Exception as e:
            print(f"[ERROR] Wav2Vec2 transcription failed: {e}")
            return False