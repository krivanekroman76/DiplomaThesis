import os
import json
import logging
import vosk
import librosa
import gc
from .utils import save_transcription_to_file

class VoskTranscription:
    def __init__(self, custom_models_dir=None):
        self.model = None
        self.current_model_name = None
        self.models_dir = custom_models_dir or os.path.join(os.getcwd(), "Models")

    def transcribe(self, audio_path, output_path, model_name, use_diarization=False, device_choice="Auto", progress_callback=None):
        prefix = "[CPU]" # Vosk runs on CPU
        try:
            if self.current_model_name != model_name:
                if progress_callback: progress_callback(5, f"{prefix} Vosk: Loading {model_name}...")
                path = os.path.join(self.models_dir, "vosk", model_name)
                self.model = vosk.Model(path)
                self.current_model_name = model_name

            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            int_audio = (audio * 32767).astype('int16').tobytes()
            
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)

            results = []
            chunk_size = 4000
            for i in range(0, len(int_audio), chunk_size):
                if rec.AcceptWaveform(int_audio[i:i+chunk_size]):
                    results.append(json.loads(rec.Result()))
                
                if progress_callback and i % 160000 == 0: # Update every 5 seconds of audio
                    percent = (i / len(int_audio)) * 80
                    progress_callback(10 + percent, f"{prefix} Vosk: Transcribing...")

            results.append(json.loads(rec.FinalResult()))
            
            # Format to unified segment style
            full_text = []
            segments = []
            for r in results:
                if "text" in r and r["text"].strip():
                    full_text.append(r["text"])
                    # Vosk gives word-level timestamps, we take the first and last for the segment
                    if "result" in r:
                        segments.append({
                            "start": r["result"][0]["start"],
                            "end": r["result"][-1]["end"],
                            "text": r["text"]
                        })

            save_transcription_to_file(output_path, model_name, full_text, segments)
            if progress_callback: progress_callback(100, f"{prefix} Vosk: Success")
            return True, os.path.basename(output_path)
        except Exception as e:
            logging.error(f"Vosk Error: {e}")
            return False, None
        finally:
            gc.collect()