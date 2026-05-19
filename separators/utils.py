import os
import torch
import torchaudio
import gc
import sys
import logging
import time
import re
import urllib.request
import zipfile
import music_tag
from pydub import AudioSegment
import tensorflow as tf
import numpy as np
from silero_vad import load_silero_vad, get_speech_timestamps, collect_chunks

# Mute chatty background libraries - DO NOT call basicConfig here
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('h5py').setLevel(logging.ERROR)
#logging.getLogger('spleeter').setLevel(logging.WARNING) 
logging.getLogger('torio').setLevel(logging.ERROR)

################################
# Separation tools functions
################################

def get_app_dir():
    """Finds the directory where the app/exe is running from."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def setup_ffmpeg_environment():
    """Injects the local ffmpeg path into the system's PATH variable temporarily."""
    import subprocess
    app_dir = get_app_dir()
    repo_root = os.path.abspath(os.path.join(app_dir, os.pardir))
    cwd = os.path.abspath(os.getcwd())
    candidate_dirs = [app_dir, repo_root]
    if cwd not in candidate_dirs:
        candidate_dirs.append(cwd)

    if os.environ.get("FFMPEG_INJECTED") != "TRUE":
        injected = False
        for candidate in candidate_dirs:
            ffmpeg_path = os.path.join(candidate, "ffmpeg.exe")
            if os.path.exists(ffmpeg_path):
                os.environ["PATH"] = candidate + os.pathsep + os.environ.get("PATH", "")
                logging.info(f"FFmpeg path injected from: {candidate}")
                injected = True
                break

        if not injected:
            logging.warning(
                "ffmpeg.exe not found in app, repository root, or current working directory! "
                "Audio processing may fail if not installed system-wide."
            )
            # Try to check if ffmpeg is available system-wide
            try:
                import subprocess
                result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logging.info("FFmpeg found system-wide.")
                else:
                    raise FileNotFoundError
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logging.error("FFmpeg is not available. Please install FFmpeg and place ffmpeg.exe in the repository root or ensure it's in PATH.")
                raise RuntimeError("FFmpeg is required for audio processing but not found.")

        os.environ["FFMPEG_INJECTED"] = "TRUE"

def get_unique_filename(base_path):
    """Generates a unique filename by appending _1, _2, etc., to avoid overwrites."""
    if not os.path.exists(base_path):
        return base_path
    base, ext = os.path.splitext(base_path)
    counter = 1
    while True:
        new_path = f"{base}_{counter}{ext}"
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def resolve_tensorflow_device(device_choice: str):
    """
    Returns 'GPU' or 'CPU' for Spleeter/TensorFlow logic.
    Also handles the environment variable to force CPU if needed.
    """
    device_choice = device_choice.upper()
    
    # Check physical hardware
    gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
    
    if device_choice in ["AUTO", "GPU", "CUDA"] and gpu_available:
        # TensorFlow generally handles GPU placement automatically, 
        # but we ensure the environment isn't masking it.
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        return "GPU"
    
    # Force CPU by masking the GPU from TensorFlow
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    return "CPU"

def get_audio_metadata(file_path):
    """
    Metadata extraction with a robust multi-tiered approach: 
    1. music-tag (Robust, supports FLAC/MP3/WAV)
    2. Pydub/FFmpeg (Fallback)
    """
    tags = {}
    try:
        # Tier 1: Specialized Tagging Library
        f = music_tag.load_file(file_path)
        
        # Pylance fix: Explicitly check if 'f' was loaded successfully
        if f is not None:
            standard_fields = ['title', 'artist', 'album', 'year', 'tracknumber', 'genre', 'comment']
            for field in standard_fields:
                # Use dictionary-style access but check if the resulting tag object is valid
                tag_obj = f[field]
                if tag_obj and tag_obj.value:
                    tags[field] = str(tag_obj.value)
        
        if tags:
            return tags
            
    except Exception as e:
        logging.debug(f"music-tag extraction failed for {file_path}: {e}")

    try:
        # Tier 2: Fallback to original Pydub method
        audio = AudioSegment.from_file(file_path)
        if hasattr(audio, 'tags') and audio.tags:
            return dict(audio.tags)
    except Exception as e:
        logging.warning(f"Total metadata extraction failure: {e}")
        
    return {}

def prepare_stem_metadata(original_tags, stem_type="Vocals"):
    """
    Modifies the content. Handles both 'title' and 'Title' keys 
    to be extra safe.
    """
    if not original_tags: 
        return {}
        
    new_tags = original_tags.copy()
    
    # Check for 'title' in any case
    target_key = None
    for k in new_tags.keys():
        if k.lower() == 'title':
            target_key = k
            break
            
    if target_key:
        new_tags[target_key] = f"{new_tags[target_key]} ({stem_type})"
    else:
        new_tags['title'] = f"Separated {stem_type}"
        
    return new_tags

def finalize_metadata(tags, stem_type, tool_name="Separator"):
    """
    The Technical Sanitizer: Final pass for FFmpeg compatibility.
    """
    if not tags or len(tags) == 0:
        print(f"INFO: [{tool_name}] No metadata found for {stem_type}. Exporting clean.")
        return None
    
    # Ensure all keys are lowercase and all values are strings
    return {str(k).lower(): str(v) for k, v in tags.items()}

################################
# Transcription tools functions
################################

def clear_memory_cache():
    """Forces garbage collection and clears PyTorch VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()

def save_transcription_to_file(output_path, model_name, text_blocks, segments):
    """Unified file saver supporting Whisper, Wav2Vec2, and Vosk formatting."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Transcription (Model: {model_name}):\n")
        f.write(" ".join(text_blocks) + "\n\n")
        f.write("Timestamps:\n")
        for seg in segments:
            # Handles Vosk/Whisper dictionaries
            start = seg.get('start', 0.0)
            end = seg.get('end', 0.0)
            text = seg.get('text', '').strip()
            speaker = seg.get('speaker', None)
            
            if speaker:
                f.write(f"{start:06.2f}s - {end:06.2f}s [{speaker}]: {text}\n")
            else:
                f.write(f"{start:.2f}s - {end:.2f}s: {text}\n")

def apply_vad(input_path: str, output_path: str) -> str:
    """
    Applies Silero VAD to an audio file, removing pure silence and instrumentals.
    Returns the path to the VAD-cleaned audio. If VAD fails or finds no speech, 
    it safely returns the original input path.
    """
    try:
        # 1. Load the Silero VAD model using the modern API
        model = load_silero_vad()

        # 2. Read audio. Silero requires exactly 16000 Hz
        wav, sample_rate = torchaudio.load(input_path)
        
        # Convert to mono if it's stereo
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            wav = resampler(wav)
            
        # Squeeze to 1D array for Silero
        wav = wav.squeeze(0)

        # 3. Get timestamps of actual speech
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000)

        # 4. If speech is found, stitch the active chunks together and save
        if len(speech_timestamps) > 0:
            cleaned_wav = collect_chunks(speech_timestamps, wav)
            # Add unsqueeze to make it 2D (1, samples) for saving
            torchaudio.save(output_path, cleaned_wav.unsqueeze(0), 16000)
            return output_path
        else:
            logging.warning(f"VAD detected no speech in {input_path}. Falling back to original audio.")
            return input_path

    except Exception as e:
        logging.error(f"VAD Processing failed for {input_path}: {e}")
        return input_path # Safe fallback
     
######################
# Both tools functions
######################

def resolve_torch_device(device_choice: str, return_string=False):
    """Returns the optimal PyTorch device based on user choice and hardware availability."""
    target = "cpu"
    if device_choice.upper()in ["AUTO", "GPU", "CUDA"]:
        if torch.cuda.is_available():
            target = "cuda"
        elif hasattr(torch.backends, 'MPS') and torch.backends.mps.is_available():
            target = "mps"
    
    return target if return_string else torch.device(target)

def download_required_models(models_dir, tool_filter=None, demucs_models=None, 
                             whisper_models=None, wav2vec2_models=None, status_callback=None):
    """
    Downloads models into the specified models_dir.
    
    Args:
        models_dir (str): The absolute path to where models should be saved.
        tool_filter (str): If provided (e.g., "Demucs"), only downloads that specific tool.
        demucs_models (list): List of Demucs model names.
        whisper_models (list): List of Whisper model names (e.g., ["base", "small"]).
        wav2vec2_models (list): List of HuggingFace Wav2Vec2 model IDs.
        status_callback (callable): Function to report status back to GUI or console.
    """
    status_report = {}

    def report(tool, status):
        status_report[tool] = status
        if status_callback:
            status_callback(tool, status)

    # 1. SPLEETER
    if tool_filter is None or tool_filter == "Spleeter":
        try:
            os.environ["MODEL_PATH"] = models_dir
            spleeter_check = os.path.join(models_dir, "2stems")
            if os.path.exists(spleeter_check):
                report("Spleeter", "Found 🔍")
            else:
                from spleeter.separator import Separator
                Separator('spleeter:2stems')
                report("Spleeter", "Downloaded ✅")
        except Exception as e:
            report("Spleeter", f"Error ❌ ({e})")

    # 2. DEMUCS
    if tool_filter is None or tool_filter == "Demucs":
        if demucs_models:
            try:
                import demucs.pretrained
                os.environ["TORCH_HOME"] = models_dir
                for m in demucs_models:
                    demucs.pretrained.get_model(m)
                report("Demucs", "Downloaded/Found ✅")
            except Exception as e:
                report("Demucs", f"Error ❌ ({e})")

    # 3. OPENUNMIX
    if tool_filter is None or tool_filter == "OpenUnmix":
        try:
            import torch
            os.environ["TORCH_HOME"] = models_dir
            hub_dir = os.path.join(models_dir, "hub")
            
            if os.path.exists(hub_dir) and any("open-unmix" in d.lower() for d in os.listdir(hub_dir)):
                report("OpenUnmix", "Found 🔍")
            else:
                torch.hub.load('sigsep/open-unmix-pytorch', 'umxhq', trust_repo=True)
                report("OpenUnmix", "Downloaded ✅")
        except Exception as e:
            report("OpenUnmix", f"Error ❌ ({e})")

    # 4. WHISPER (Fixed: Routed to /whisper subfolder, avoids RAM load)
    if tool_filter is None or tool_filter == "Whisper":
        if whisper_models:
            try:
                import whisper
                whisper_dir = os.path.join(models_dir, "whisper")
                os.makedirs(whisper_dir, exist_ok=True)
                
                for m in whisper_models:
                    # Using _download directly prevents the model from loading into RAM/VRAM during the fetch
                    whisper._download(whisper._MODELS[m], whisper_dir, False)
                report("Whisper", "Downloaded/Found ✅")
            except Exception as e:
                report("Whisper", f"Error ❌ ({e})")

    # 5. WAV2VEC2 (Fixed: Routed HuggingFace cache to /huggingface subfolder)
    if tool_filter is None or tool_filter == "Wav2Vec2":
        if wav2vec2_models:
            try:
                # Redirect HuggingFace cache strictly to our huggingface subfolder
                hf_dir = os.path.join(models_dir, "huggingface")
                os.environ["HF_HOME"] = hf_dir
                os.environ["TRANSFORMERS_CACHE"] = hf_dir
                
                from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
                
                for m in wav2vec2_models:
                    Wav2Vec2Processor.from_pretrained(m)
                    Wav2Vec2ForCTC.from_pretrained(m)
                report("Wav2Vec2", "Downloaded/Found ✅")
            except Exception as e:
                report("Wav2Vec2", f"Error ❌ ({e})")

    # 6. VOSK
    if tool_filter is None or tool_filter == "Vosk":
        try:
            # Default small English and French models
            vosk_models = ["vosk-model-small-en-us-0.15", "vosk-model-small-fr-0.22"]
            vosk_dir = os.path.join(models_dir, "vosk")
            os.makedirs(vosk_dir, exist_ok=True)
            
            for m in vosk_models:
                model_path = os.path.join(vosk_dir, m)
                if os.path.exists(model_path):
                    continue
                
                logging.info(f"Downloading Vosk model: {m}...")
                url = f"https://alphacephei.com/vosk/models/{m}.zip"
                zip_path = model_path + ".zip"
                
                urllib.request.urlretrieve(url, zip_path)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(vosk_dir)
                os.remove(zip_path) # Clean up the zip file
                
            report("Vosk", "Downloaded/Found ✅")
            
            # Advice for the user on adding more models
            logging.info("\n" + "="*50)
            logging.info("💡 HOW TO ADD MORE VOSK MODELS:")
            logging.info("1. Visit: https://alphacephei.com/vosk/models")
            logging.info("2. Download the .zip file for your desired language.")
            logging.info(f"3. Extract the folder inside the zip into: {vosk_dir}")
            logging.info(f"   (e.g., {os.path.join(vosk_dir, 'vosk-model-de-0.21')})")
            logging.info("="*50 + "\n")
            
        except Exception as e:
            report("Vosk", f"Error ❌ ({e})")

    return status_report

class ProgressInterceptor(logging.Handler):
    def __init__(self, callback, device="CPU", tool_name="Process", total_duration=0):
        super().__init__()
        self.callback = callback
        self.device = str(device).upper()
        self.tool_name = tool_name
        self.total_duration = total_duration  # Received from librosa in the main script
        self.current_action = "Initializing"
        logging.info(f"Duration passed: {self.total_duration}")

    def emit(self, record):
        log_entry = record.getMessage()
        lower_entry = log_entry.lower()
        
        # --- 1. CLEANUP: Suppress Whisper/Library "False Errors" ---
        # These are strings that libraries log as ERROR but are actually INFO
        suppress_keywords = [
            "-->", "config.json", "weights of Wav2Vec2", 
            "decoding params", "voskapi", "detected language"
        ]
        
        if any(k in lower_entry for k in suppress_keywords):
            record.levelno = logging.INFO
            record.levelname = "INFO"

        if not self.callback:
            return

        # --- 2. WHISPER PROGRESS (Timestamp Based) ---
        # Matches: [01:22.000 --> 01:25.000]
        timestamp_match = re.search(r'\[(\d{2}):(\d{2})\.\d{3}\s+-->', log_entry)
        
        if timestamp_match and self.total_duration > 0:
            mins, secs = int(timestamp_match.group(1)), int(timestamp_match.group(2))
            current_seconds = (mins * 60) + secs
            
            # Map progress to 10% - 90% range to leave room for loading/saving
            raw_pct = (current_seconds / self.total_duration) * 80
            progress_val = int(raw_pct + 10)
            
            # Ensure we don't exceed 95% purely from timestamps
            progress_val = min(95, progress_val)
            
            display_text = f"[{self.device}] {self.tool_name}: {progress_val}% ({mins:02d}:{secs:02d})"
            self.callback(progress_val, display_text)
            return # Exit early if we handled a timestamp

        # --- 3. GENERAL PROGRESS (Percentage Based) ---
        # Matches standard "55%" logs from Demucs or other tools
        pct_match = re.search(r'(\d{1,3})%', log_entry)
        if pct_match:
            val = int(pct_match.group(1))
            self.callback(val, f"[{self.device}] {self.tool_name} Processing: {val}%")
            return

        # --- 4. ACTION DETECTION (Spleeter/General) ---
        if "loading" in lower_entry or "weights" in lower_entry:
            self.callback(15, f"[{self.device}] {self.tool_name}: Loading Models...")
        
        elif "writing" in lower_entry or "written" in lower_entry:
            self.callback(90, f"[{self.device}] {self.tool_name}: Saving Output...")