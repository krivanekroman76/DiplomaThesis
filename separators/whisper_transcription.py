import os
import whisper

class WhisperTranscription:
    def __init__(self):
       self.loaded_models = {}  # Cache loaded models to avoid reloading

    def load_model(self, model_name: str):
        """Load and cache the Whisper model if not already loaded."""
        if model_name not in self.loaded_models:
            try:
                print(f"Whisper: Loading model '{model_name}'...")
                self.loaded_models[model_name] = whisper.load_model(model_name)
                print(f"Whisper: Model '{model_name}' loaded successfully.")
            except Exception as e:
                raise ValueError(f"Failed to load Whisper model '{model_name}': {e}")
        return self.loaded_models[model_name]

    def transcribe(self, audio_path: str, output_path: str, model_name: str = "base", verbose: bool = False):
        """
        Transcribe the audio file using the specified Whisper model and save to output_path.
       
        :param audio_path: Path to the audio file (e.g., vocals.wav).
        :param output_path: Path to save the transcription (e.g., transcription.txt).
        :param model_name: Whisper model name (e.g., "tiny", "base", "small", "medium", "large", "turbo").
        :param verbose: If True, enable verbose output during transcription.
        :return: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(audio_path):
               raise FileNotFoundError(f"Audio file not found: {audio_path}")
           
            # Load the model
            model = self.load_model(model_name)
               
            # Perform transcription
            print(f"Whisper: Transcribing '{audio_path}' with model '{model_name}'...")
            result = model.transcribe(audio_path, verbose=verbose)
               
            # Write the transcription to file
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Transcription (Model: {model_name}):\n{result['text']}\n\n")
                
                # Optional: Add timestamps if available
                if "segments" in result:
                    f.write("Timestamps:\n")
                    for seg in result["segments"]:
                        f.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text']}\n")
               
            print(f"Whisper: Transcription saved to '{output_path}'.")
            return True
           
        except Exception as e:
           print(f"Whisper transcription error: {e}")
           return False
   