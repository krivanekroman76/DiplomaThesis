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

    def _get_unique_filename(self, base_path):
        """
        Overview: Generate a unique filename by appending _1, _2, etc., if the file exists.
        
        Parameters:
        - base_path (str): The initial file path to check.
        
        Returns:
        - str: A unique file path that doesn't exist.
        """
        if not os.path.exists(base_path):
            return base_path
        base, ext = os.path.splitext(base_path)
        counter = 1
        while True:
            new_path = f"{base}_{counter}{ext}"
            if not os.path.exists(new_path):
                return new_path
            counter += 1

    def transcribe(self, audio_path, output_path, model_name, language="auto"):
        """
        Transcribe the audio file using the specified Whisper model and save to output_path.
       
        :param audio_path: Path to the audio file (e.g., vocals.wav).
        :param output_path: Path to save the transcription.
        :param model_name: Whisper model name (e.g., "tiny", "base", "small", "medium", "large", "turbo").
        :return: True if successful, False otherwise.
        """
        try:
            if not os.path.exists(audio_path):
               raise FileNotFoundError(f"Audio file not found: {audio_path}")
           
            # Load the model
            model = self.load_model(model_name)
            
            # If language is "auto", pass None to whisper, otherwise pass the lang code
            lang_param = None if language == "auto" else language
       
            # Perform transcription
            print(f"Whisper: Transcribing '{audio_path}' with model '{model_name}'...")
            result = self.model.transcribe(audio_path, language=lang_param)

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
   