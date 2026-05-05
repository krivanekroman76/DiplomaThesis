import os
import sys
import json
import logging
import numpy as np
from scipy.spatial.distance import cosine
import vosk
import librosa
import gc
from .utils import save_transcription_to_file

vosk.SetLogLevel(-1)

class VoskTranscription:
    def __init__(self, custom_models_dir=None):
        self.loaded_models = {}
        self.spk_model = None
        
        # --- THE FIX: Check if frozen (compiled) or running from source ---
        if getattr(sys, 'frozen', False):
            # If compiled to .exe, base directory is where the .exe lives
            base_project_dir = os.path.dirname(sys.executable)
        else:
            # If running from script, go up one directory from the /separators folder
            base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.models_dir = custom_models_dir if custom_models_dir else os.path.join(base_project_dir, "Models")
        
        if not os.path.exists(self.models_dir):
            logging.warning(f"Vosk models directory not found at: {self.models_dir}")
            os.makedirs(self.models_dir, exist_ok=True)
        
        self.speaker_profiles = {}

    def load_model(self, model_name: str, spk_model_name="vosk-model-spk-0.4"):
        if model_name not in self.loaded_models:
            # --- We added "vosk" to both of these paths! ---
            model_path = os.path.join(self.models_dir, "vosk", model_name)
            spk_path = os.path.join(self.models_dir, "vosk", spk_model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Vosk: Model not found at {model_path}")
            
            logging.info(f"Vosk: Loading language model '{model_name}'...")
            self.loaded_models[model_name] = vosk.Model(model_path)
            
            if os.path.exists(spk_path):
                logging.info(f"Vosk: Loading Speaker Model from '{spk_path}'...")
                self.spk_model = vosk.SpkModel(spk_path)
            else:
                logging.warning("Vosk: Speaker model not found. Diarization disabled.")
        
        return self.loaded_models[model_name]

    def identify_speaker(self, new_vector, threshold=0.25):
        if not new_vector: return "Unknown"
        
        best_speaker = None
        min_dist = threshold

        for speaker_id, vectors in self.speaker_profiles.items():
            dist = np.mean([cosine(new_vector, v) for v in vectors])
            if dist < min_dist:
                min_dist = dist
                best_speaker = speaker_id

        if best_speaker is None:
            new_id = f"Vocal {len(self.speaker_profiles) + 1}"
            self.speaker_profiles[new_id] = [new_vector]
            return new_id
        
        self.speaker_profiles[best_speaker].append(new_vector)
        return best_speaker

    def transcribe(self, audio_path, output_path, model_name, use_diarization=False, device_choice="Auto", progress_callback=None):
        try:
            if progress_callback: progress_callback(5, f"Vosk: Loading model '{model_name}'...")
            model = self.load_model(model_name)
            
            if progress_callback: progress_callback(15, "Vosk: Preparing audio...")
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            int_audio = (audio * 32767).astype('int16').tobytes()

            if use_diarization and self.spk_model:
                rec = vosk.KaldiRecognizer(model, 16000, self.spk_model)
            else:
                rec = vosk.KaldiRecognizer(model, 16000)
            rec.SetWords(True)

            results = []
            chunk_size = 8000
            total_bytes = len(int_audio)
            total_chunks = max(1, total_bytes // chunk_size)
            self.speaker_profiles = {} 

            for chunk_idx, i in enumerate(range(0, total_bytes, chunk_size)):
                if progress_callback and chunk_idx % 20 == 0:  
                    percent_done = (chunk_idx / total_chunks) * 75
                    progress_callback(15 + percent_done, f"Vosk: Transcribing... ({int((chunk_idx/total_chunks)*100)}%)")

                chunk = int_audio[i:i+chunk_size]
                if rec.AcceptWaveform(chunk):
                    results.append(json.loads(rec.Result()))
            
            results.append(json.loads(rec.FinalResult()))

            if progress_callback: progress_callback(90, "Vosk: Formatting output...")

            full_text_blocks = []
            all_segments = []

            for res in results:
                if "result" in res and "text" in res and res["text"].strip() != "":
                    text = res["text"]
                    full_text_blocks.append(text)
                    
                    speaker_label = "Vocal"
                    if use_diarization and "spk" in res:
                        speaker_label = self.identify_speaker(res.get("spk"))
                    
                    all_segments.append({
                        "start": res["result"][0]["start"],
                        "end": res["result"][-1]["end"],
                        "speaker": speaker_label,
                        "text": text
                    })

            if progress_callback: progress_callback(95, "Vosk: Saving file...")

            save_transcription_to_file(output_path, model_name, full_text_blocks, all_segments)

            del audio
            del int_audio
            gc.collect()

            logging.info(f"Vosk: Transcription saved to '{output_path}'.")
            if progress_callback: progress_callback(100, "Vosk: Complete!")
            return True, os.path.basename(output_path)

        except Exception as e:
            logging.error(f"Vosk Error: {e}", exc_info=True)
            if progress_callback: progress_callback(0, f"Error: {str(e)}")
            return False, None