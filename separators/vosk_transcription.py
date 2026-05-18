import os
import json
import logging
import vosk
import librosa
import gc
from .utils import (
    save_transcription_to_file, 
    ProgressInterceptor,
    apply_vad # --- NEW IMPORT ---
)

class VoskTranscription:
    def __init__(self, custom_models_dir=None):
        self.model = None
        self.current_model_name = None
        self.models_dir = custom_models_dir or os.path.join(os.getcwd(), "Models")

    def transcribe(self, audio_path, output_path, model_name, use_diarization=False, device_choice="Auto", progress_callback=None):
        prefix = "[CPU]" 
        
        # --- APPLY VAD STEP ---
        vad_output_path = audio_path.replace(".wav", "_vad.wav")
        if progress_callback:
            progress_callback(2, "Applying Voice Activity Detection...")
        
        processed_audio_path = apply_vad(audio_path, vad_output_path)
        # ----------------------

        # 1. Setup Progress Interceptor
        total_duration = librosa.get_duration(filename=processed_audio_path)
        handler = ProgressInterceptor(progress_callback, device="CPU", tool_name="Vosk", total_duration=int(total_duration))
        logging.getLogger().addHandler(handler)

        try:
            # 2. Load Model (Progress 5-10%)
            if self.current_model_name != model_name:
                if progress_callback: progress_callback(5, f"{prefix} Vosk: Loading model '{model_name}'...")
                path = os.path.join(self.models_dir, "vosk", model_name)
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Vosk model not found at: {path}")
                self.model = vosk.Model(path)
                self.current_model_name = model_name

            # 3. Audio Loading
            audio, _ = librosa.load(processed_audio_path, sr=16000, mono=True)
            int_audio = (audio * 32767).astype('int16').tobytes()
            
            rec = vosk.KaldiRecognizer(self.model, 16000)
            rec.SetWords(True)

            results = []
            chunk_size = 4000 # ~250ms of audio
            
            # 4. Transcription Loop (Progress 10-90%)
            total_bytes = len(int_audio)
            for i in range(0, total_bytes, chunk_size):
                chunk = int_audio[i:i+chunk_size]
                if rec.AcceptWaveform(chunk):
                    results.append(json.loads(rec.Result()))
                
                # Update UI every ~5 seconds of audio to keep it smooth
                if progress_callback and i % 80000 == 0: 
                    percent = (i / total_bytes) * 80
                    progress_callback(int(10 + percent), f"{prefix} Vosk: Transcribing audio...")

            results.append(json.loads(rec.FinalResult()))
            
            # 5. Format and Save (Progress 95-100%)
            full_text = []
            segments = []
            for r in results:
                if "text" in r and r["text"].strip():
                    full_text.append(r["text"])
                    if "result" in r and len(r["result"]) > 0:
                        segments.append({
                            "start": r["result"][0]["start"],
                            "end": r["result"][-1]["end"],
                            "text": r["text"]
                        })

            if progress_callback: progress_callback(95, f"{prefix} Vosk: Saving text file...")
            save_transcription_to_file(output_path, model_name, full_text, segments)
            
            if progress_callback: progress_callback(100, f"{prefix} Vosk: Success!")
            return True, os.path.basename(output_path)

        except Exception as e:
            logging.error(f"Vosk Error: {e}")
            return False, None
        finally:
            logging.getLogger().removeHandler(handler)
            gc.collect()
            # --- CLEANUP TEMP VAD FILE ---
            if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                try:
                    os.remove(processed_audio_path)
                except Exception as cleanup_error:
                    logging.warning(f"Failed to delete temp VAD file: {cleanup_error}")