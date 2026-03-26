import os
import json
import numpy as np
from scipy.spatial.distance import cosine
import vosk
import librosa
import gc

# Add this line right here to silence the C++ console spam!
vosk.SetLogLevel(-1)

class VoskTranscription:
    def __init__(self, custom_models_dir=None):
        # Cache for loaded ASR models to avoid redundant disk I/O
        self.loaded_models = {}
        
        # Placeholder for the speaker identification (diarization) model
        self.spk_model = None
        
        # Dynamically resolve the project base directory
        base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Use the provided custom directory, or default to the centralized 'Models' folder
        if custom_models_dir:
            self.models_dir = custom_models_dir
        else:
            self.models_dir = os.path.join(base_project_dir, "Models")
        
        if not os.path.exists(self.models_dir):
            print(f"[WARNING] Vosk models directory not found at: {self.models_dir}")
            os.makedirs(self.models_dir, exist_ok=True)
        else:
            print(f"[INFO] Vosk models directory set to: {self.models_dir}")
        
        # Dictionary to store unique voice vectors for different speakers
        self.speaker_profiles = {}

    def load_model(self, model_name: str, spk_model_name="vosk-model-spk-0.4"):
        """Loads the main language model and the optional speaker model."""
        if model_name not in self.loaded_models:
            model_path = os.path.join(self.models_dir, model_name)
            spk_path = os.path.join(self.models_dir, spk_model_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Vosk: Model not found at {model_path}")
            
            print(f"[INFO] Vosk: Loading language model '{model_name}'...")
            self.loaded_models[model_name] = vosk.Model(model_path)
            
            # Load speaker model for diarization if available
            if os.path.exists(spk_path):
                print(f"[INFO] Vosk: Loading Speaker Model from '{spk_path}'...")
                self.spk_model = vosk.SpkModel(spk_path)
            else:
                print("[WARNING] Vosk: Speaker model not found. Diarization disabled.")
        
        return self.loaded_models[model_name]

    def identify_speaker(self, new_vector, threshold=0.25):
        """Assigns a label (Vocal 1, 2...) based on cosine similarity of voice vectors."""
        if not new_vector: 
            return "Unknown"
        
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
        else:
            self.speaker_profiles[best_speaker].append(new_vector)
            return best_speaker

    def transcribe(self, audio_path, output_path, model_name, use_diarization=False, device_choice="Auto", progress_callback=None):
        """Perform transcription with optional speaker diarization and Whisper-style formatting."""
        try:
            if progress_callback:
                progress_callback(5, f"Vosk: Loading model '{model_name}'...")

            model = self.load_model(model_name)
            
            if progress_callback:
                progress_callback(15, "Vosk: Preparing audio...")
            
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

            # Reset speaker profiles for each new song to avoid cross-contamination
            self.speaker_profiles = {} 

            # Process audio in chunks to generate a smooth progress bar
            for chunk_idx, i in enumerate(range(0, total_bytes, chunk_size)):
                
                # Update UI every 20 chunks so we don't overwhelm the main thread
                if progress_callback and chunk_idx % 20 == 0:  
                    percent_done = (chunk_idx / total_chunks) * 75
                    progress_callback(
                        15 + percent_done, 
                        f"Vosk: Transcribing... ({int((chunk_idx/total_chunks)*100)}%)"
                    )

                chunk = int_audio[i:i+chunk_size]
                if rec.AcceptWaveform(chunk):
                    results.append(json.loads(rec.Result()))
            
            results.append(json.loads(rec.FinalResult()))

            if progress_callback:
                progress_callback(90, "Vosk: Formatting output...")

            # --- Format data exactly like Whisper ---
            full_text_blocks = []
            all_segments = []

            for res in results:
                if "result" in res and "text" in res and res["text"].strip() != "":
                    text = res["text"]
                    full_text_blocks.append(text)
                    
                    # Identification Logic
                    speaker_label = "Vocal"
                    if use_diarization and "spk" in res:
                        speaker_label = self.identify_speaker(res.get("spk"))
                    
                    start_t = res["result"][0]["start"]
                    end_t = res["result"][-1]["end"]
                    
                    all_segments.append({
                        "start": start_t,
                        "end": end_t,
                        "speaker": speaker_label,
                        "text": text
                    })

            if progress_callback:
                progress_callback(95, "Vosk: Saving file...")

            # --- Save to file ---
            with open(output_path, "w", encoding="utf-8") as f:
                # Write the full block of text
                f.write(f"Transcription (Model: {model_name}):\n")
                f.write(" ".join(full_text_blocks) + "\n\n")
                
                # Write the Timestamps with Speaker Tags
                f.write("Timestamps:\n")
                for seg in all_segments:
                    f.write(f"{seg['start']:06.2f}s - {seg['end']:06.2f}s [{seg['speaker']}]: {seg['text']}\n")

            # --- Clean up RAM ---
            del audio
            del int_audio
            gc.collect()

            print(f"[INFO] Vosk: Transcription saved to '{output_path}'.")
            
            if progress_callback:
                progress_callback(100, "Vosk: Complete!")

            # Return the success boolean and filename to the GUI thread
            return True, os.path.basename(output_path)

        except Exception as e:
            print(f"[ERROR] Vosk transcription error: {e}")
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")
            return False, None