import os
import whisper
import librosa
import math
import torch
import gc

class WhisperTranscription:
    def __init__(self):
        # We only ever keep ONE model loaded at a time to protect RAM/VRAM
        self.current_model_name = None
        self.model = None

    def load_model(self, model_name: str):
        """Load the Whisper model, safely dumping the old one if needed."""
        if model_name != self.current_model_name:
            # 1. Clear old model from memory
            if self.model is not None:
                del self.model
                self.model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            
            # 2. Define the custom path
            import sys
            base_dir = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.abspath(".")
            whisper_path = os.path.join(base_dir, "pretrained_models", "whisper")
            os.makedirs(whisper_path, exist_ok=True)
            
            # 3. Load new model
            print(f"[INFO] Whisper: Loading model '{model_name}'...")
            # Tell Whisper EXACTLY where to look/download
            self.model = whisper.load_model(model_name, download_root=whisper_path)
            self.current_model_name = model_name

    def transcribe(self, audio_path, output_path, model_name, language="auto", progress_callback=None):
        """
        Transcribes audio in RAM-safe chunks with real-time progress updates.
        Maintains your beautiful Text + Timestamps output format.
        """
        try:
            if progress_callback:
                progress_callback(5, f"Whisper: Initializing '{model_name}'...")

            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # 1. Safely load the model
            self.load_model(model_name)
            
            # 2. Load the audio into a mathematical array
            if progress_callback:
                progress_callback(15, "Whisper: Preparing audio chunks...")
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # 3. Setup Chunking Math (30 second chunks)
            chunk_length_s = 30
            chunk_samples = chunk_length_s * 16000
            total_samples = len(audio)
            total_chunks = math.ceil(total_samples / chunk_samples)

            full_text_blocks = []
            all_segments = []
            previous_context = ""
            lang_param = None if language == "auto" else language

            # 4. Transcribe chunk by chunk!
            for chunk_idx, i in enumerate(range(0, total_samples, chunk_samples)):
                # --- PROGRESS UPDATE ---
                if progress_callback:
                    percent_done = (chunk_idx / total_chunks) * 80
                    progress_callback(
                        15 + percent_done, 
                        f"Whisper: Transcribing chunk {chunk_idx + 1} of {total_chunks}..."
                    )
                # -----------------------

                chunk = audio[i : i + chunk_samples]
                
                # We pass the previous text as a prompt so Whisper doesn't lose the context!
                result = self.model.transcribe(
                    chunk, 
                    language=lang_param, 
                    initial_prompt=previous_context
                )

                text = result["text"].strip()
                if text:
                    full_text_blocks.append(text)
                    # Save the last 200 characters to feed into the next chunk
                    previous_context = text[-200:] 

                # Adjust the timestamps so they reflect the whole file, not just the 30s chunk
                time_offset = chunk_idx * chunk_length_s
                for seg in result["segments"]:
                    seg["start"] += time_offset
                    seg["end"] += time_offset
                    all_segments.append(seg)

            if progress_callback:
                progress_callback(95, "Whisper: Formatting and saving file...")

            # 5. Save using your beautiful custom format
            with open(output_path, "w", encoding="utf-8") as f:
                # Write Full Text
                f.write(f"Transcription (Model: {model_name}):\n")
                f.write(" ".join(full_text_blocks) + "\n\n")
                
                # Write Timestamps
                f.write("Timestamps:\n")
                for seg in all_segments:
                    f.write(f"{seg['start']:.2f}s - {seg['end']:.2f}s: {seg['text'].strip()}\n")

            # 6. Clean up temporary audio arrays
            del audio
            gc.collect()

            print(f"[INFO] Whisper: Transcription saved to '{output_path}'.")
            
            # Return True and the filename so the GUI can display it
            return True, os.path.basename(output_path)
            
        except Exception as e:
            print(f"[ERROR] Whisper transcription error: {e}")
            if progress_callback:
                progress_callback(0, f"Whisper Error: {str(e)}")
            return False, None