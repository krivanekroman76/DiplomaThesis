import os
import stt

class CoquiTranscription:
    def __init__(self, model_path="models/coqui/model.pbmm", scorer_path="models/coqui/model.scorer"):
        self.model_path = model_path
        self.scorer_path = scorer_path
        self.model = None
        self.load_model()

    def load_model(self):
        """Load the Coqui STT model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Coqui model not found at '{self.model_path}'. Download from https://coqui.ai/models and place in 'models/coqui/'.")
        self.model = stt.Model(self.model_path)
        if os.path.exists(self.scorer_path):
            self.model.enableExternalScorer(self.scorer_path)
        print(f"Coqui: Model loaded from '{self.model_path}'.")

    def transcribe(self, audio_path: str, output_path: str, model_name: str, verbose: bool = False):
        """
        Transcribe the audio file using Coqui STT and save to output_path.
        
        :param audio_path: Path to the audio file (e.g., vocals.wav).
        :param output_path: Path to save the transcription (e.g., transcription.txt).
        :param model_name: Not used (placeholder).
        :param verbose: If True, enable verbose output.
        :return: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Transcribe
            transcription = self.model.stt(stt.read_audio_file(audio_path))
            
            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription (Coqui STT):\n{transcription}\n")
            
            print(f"Coqui: Transcription saved to '{output_path}'.")
            return True
        
        except Exception as e:
            print(f"Coqui transcription error: {e}")
            return False