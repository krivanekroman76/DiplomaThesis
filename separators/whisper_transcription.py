import os
import whisper
import librosa
import logging
import torch
from .utils import (
    resolve_torch_device, 
    clear_memory_cache, 
    save_transcription_to_file, 
    ProgressInterceptor,
    apply_vad # --- NEW IMPORT ---
)

class WhisperTranscription:
    def __init__(self, custom_models_dir=None):
        self.current_model_name = None
        self.model = None
        self.models_dir = custom_models_dir or os.path.join(os.getcwd(), "Models")

    def get_style_prompt(self, language, genre_hint=""):
        """
        Generates a dynamic prompt synced with UI language keys.
        """
        # 1. Base language anchors - Synced with your UI values: ["auto", "cs", "en", "fr", "de", "es"]
        base_anchors = {
            "en": "English song lyrics, transcription, punctuation.",
            "cs": "Český text, písňové texty, diakritika, interpunkce.",
            "sk": "Slovenský text, piesňové texty, diakritika, interpunkcia.", # Added just in case
            "fr": "Paroles de chanson en français, transcription, ponctuation.",
            "de": "Deutsche Songtexte, Transkription, Satzzeichen.",
            "es": "Letras de canciones en español, transcripción, puntuación."
        }

        # 2. Genre-specific keywords - Synced with UI: ["Rap/Hip-Hop", "Rock/Metal", "Pop/Ballad", "Speech/Other"]
        genre_keywords = {
            "Rap/Hip-Hop": "Rap music, fast flow, slang, street language.",
            "Rock/Metal": "Rock music, energetic vocals, lyrics.",
            "Pop/Ballad": "Pop music, melodic singing, clean lyrics.",
            "Speech/Other": "Clear speech, spoken word, no background music."
        }

        # Get base anchor (default to English if not found)
        base = base_anchors.get(language, base_anchors["en"])
        
        # Get genre addition
        style = genre_keywords.get(genre_hint, "")

        # Combine them cleanly
        if style:
            return f"{base} Style: {style}"
        return base

    def transcribe(self, audio_path, output_path, model_name, language="auto", 
                  genre_hint="Pop/Ballad", device_choice="Auto", progress_callback=None):
        
        # --- APPLY VAD STEP ---
        vad_output_path = audio_path.replace(".wav", "_vad.wav")
        if progress_callback:
            progress_callback(2, "Applying Voice Activity Detection...")
        
        processed_audio_path = apply_vad(audio_path, vad_output_path)
        # ----------------------

        target_device = resolve_torch_device(device_choice, return_string=True)
        
        try:
            # Get total duration of the CLEANED audio for percentage calculations
            total_duration = librosa.get_duration(filename=processed_audio_path)
        except Exception:
            total_duration = 0

        try:
            return self._run_inference(processed_audio_path, output_path, model_name, language, 
                                     genre_hint, target_device, progress_callback, total_duration)
        except Exception as e:
            if ("out of memory" in str(e).lower()) and target_device != "cpu":
                logging.warning(f"Whisper VRAM Full. Falling back to CPU...")
                clear_memory_cache()
                return self._run_inference(processed_audio_path, output_path, model_name, language, 
                                         genre_hint, "cpu", progress_callback, total_duration)
            else:
                logging.error(f"Whisper Error: {e}")
                return False, None
        finally:
            # --- CLEANUP TEMP VAD FILE ---
            if processed_audio_path != audio_path and os.path.exists(processed_audio_path):
                try:
                    os.remove(processed_audio_path)
                except Exception as cleanup_error:
                    logging.warning(f"Failed to delete temp VAD file: {cleanup_error}")

    def _run_inference(self, audio_path, output_path, model_name, language, genre_hint, device, progress_callback, total_duration):
        prefix = f"[{str(device).upper()}]"
        
        root_logger = logging.getLogger()
        handler = ProgressInterceptor(
            progress_callback, 
            device=device, 
            total_duration=int(total_duration),
            tool_name="Whisper"
        )
        root_logger.addHandler(handler)

        try:
            # Load model if needed
            if self.current_model_name != model_name or self.model is None:
                if progress_callback: 
                    progress_callback(5, f"{prefix} Whisper: Loading {model_name}...")
                
                whisper_path = os.path.join(self.models_dir, "whisper")
                self.model = whisper.load_model(model_name, device=device, download_root=whisper_path)
                self.current_model_name = model_name

            # Dynamic Prompt Logic
            if language.lower() == "auto":
                prompt = "" # Empty prompt allows better auto-detection
                lang_param = None
            else:
                prompt = self.get_style_prompt(language, genre_hint)
                lang_param = language

            if progress_callback: 
                progress_callback(10, f"{prefix} Whisper: Starting ({genre_hint})...")

            # Execute Transcription
            result = self.model.transcribe(
                audio_path,
                language=lang_param,
                initial_prompt=prompt,
                temperature=0.0,
                condition_on_previous_text=False,  
                no_speech_threshold=0.4,           
                logprob_threshold=-1.0,            
                verbose=True 
            )

            if progress_callback: 
                progress_callback(95, f"{prefix} Whisper: Finalizing...")
            
            full_text = [result["text"]]
            segments = result["segments"]
            save_transcription_to_file(output_path, model_name, full_text, segments)
            
            if progress_callback: 
                progress_callback(100, f"{prefix} Whisper: Complete!")
            
            return True, os.path.basename(output_path)

        finally:
            root_logger.removeHandler(handler)
            clear_memory_cache()