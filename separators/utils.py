import os
import gc
import sys
import logging
import torch

# Set up global logging format once
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Silence noisy libraries globally
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

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

################################
# Transcription tools functions
################################

def resolve_torch_device(device_choice: str, return_string=False):
    """Returns the optimal PyTorch device based on user choice and hardware availability."""
    target = "cpu"
    if device_choice in ["Auto", "GPU"]:
        if torch.cuda.is_available():
            target = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            target = "mps"
    
    return target if return_string else torch.device(target)

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