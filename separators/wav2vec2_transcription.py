import os
import torch
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

class Wav2Vec2Transcription:
    def __init__(self, model_name="facebook/wav2vec2-base-960h"):
        self.model_name = model_name
        self.processor
        self.model
        self.load_model()

    def load_model(self):
        """Load the Wav2Vec2 model and processor."""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            print(f"Wav2Vec2: Model '{self.model_name}' loaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Wav2Vec2 model: {e}. Ensure transformers and torch are installed.")

    def transcribe(self, audio_path: str, output_path: str, model_name: str, verbose: bool = False):
        """
        Transcribe the audio file using Wav2Vec2 and save to output_path.
        
        :param audio_path: Path to the audio file (e.g., vocals.wav).
        :param output_path: Path to save the transcription (e.g., transcription.txt).
        :param model_name: Not used (placeholder).
        :param verbose: If True, enable verbose output.
        :return: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load audio
            audio, rate = librosa.load(audio_path, sr=16000)  # Wav2Vec2 expects 16kHz
            
            # Process
            inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = self.model(inputs.input_values).logits
            
            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            # Write to file (basic; timestamps not directly available)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription (Wav2Vec2):\n{transcription}\n")
            
            print(f"Wav2Vec2: Transcription saved to '{output_path}'.")
            return True
        
        except Exception as e:
            print(f"Wav2Vec2 transcription error: {e}")
            return False