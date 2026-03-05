import os
import json
import numpy as np
from scipy.spatial.distance import cosine
import vosk
import librosa

class VoskTranscription:
    def __init__(self):
        # Cache for loaded ASR models to avoid redundant disk I/O
        self.loaded_models = {}
        
        # Placeholder for the speaker identification (diarization) model
        self.spk_model = None
        
        # Dynamically resolve the project base directory
        # Assumes this script is located in the 'separators' subdirectory
        base_project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Define the absolute path to the pretrained Vosk models
        self.models_dir = os.path.join(base_project_dir, "pretrained_models", "vosk")
        
        # Verify directory existence and initialize if missing
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
            
            self.loaded_models[model_name] = vosk.Model(model_path)
            
            # Load speaker model for diarization if available
            if os.path.exists(spk_path):
                print(f"Vosk: Loading Speaker Model from '{spk_path}'...")
                self.spk_model = vosk.SpkModel(spk_path)
            else:
                print("Vosk: WARNING - Speaker model not found. Diarization will be disabled.")
        
        return self.loaded_models[model_name]

    def identify_speaker(self, new_vector, threshold=0.25):
        """Assigns a label (Vocal 1, 2...) based on cosine similarity of voice vectors."""
        if not new_vector: 
            return "Unknown"
        
        best_speaker = None
        min_dist = threshold

        for speaker_id, vectors in self.speaker_profiles.items():
            # Calculate mean distance from known samples of the speaker
            dist = np.mean([cosine(new_vector, v) for v in vectors])
            if dist < min_dist:
                min_dist = dist
                best_speaker = speaker_id

        if best_speaker is None:
            # New speaker detected
            new_id = f"Vocal {len(self.speaker_profiles) + 1}"
            self.speaker_profiles[new_id] = [new_vector]
            return new_id
        else:
            # Update existing speaker profile with new vector for better accuracy
            self.speaker_profiles[best_speaker].append(new_vector)
            return best_speaker

    def transcribe(self, audio_path, output_path, model_name, use_diarization=False):
        """Perform transcription with optional speaker diarization."""
        try:
            model = self.load_model(model_name)
            audio, _ = librosa.load(audio_path, sr=16000, mono=True)
            int_audio = (audio * 32767).astype('int16').tobytes()

            # Initialize with SPK model only if requested AND available
            if use_diarization and self.spk_model:
                rec = vosk.KaldiRecognizer(model, 16000, self.spk_model)
            else:
                rec = vosk.KaldiRecognizer(model, 16000)
            
            rec.SetWords(True)

            results = []
            chunk_size = 8000
            for i in range(0, len(int_audio), chunk_size):
                if rec.AcceptWaveform(int_audio[i:i+chunk_size]):
                    results.append(json.loads(rec.Result()))
            results.append(json.loads(rec.FinalResult()))

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Karaoke Transcript (Vosk):\n\n")
                
                # Reset speaker profiles for each new song to avoid cross-contamination
                self.speaker_profiles = {} 

                for res in results:
                    if "result" in res and "text" in res and res["text"].strip() != "":
                        # Identification Logic
                        if use_diarization and "spk" in res:
                            speaker_label = self.identify_speaker(res.get("spk"))
                        else:
                            speaker_label = "Vocal"

                        start_t = res["result"][0]["start"]
                        end_t = res["result"][-1]["end"]
                        
                        line = f"[{start_t:06.2f} - {end_t:06.2f}] {speaker_label}: {res['text']}\n"
                        f.write(line)
                        print(line.strip())

            return True
        except Exception as e:
            print(f"Vosk transcription error: {e}")
            return False