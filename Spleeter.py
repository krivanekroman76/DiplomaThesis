import os
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter

def separate_S(input_path, vocals_out_dir, instr_out_dir, trans_out_dir, transcribe, ai_suffix="_S"):
    """
    Separate audio using Spleeter 2 stems model.

    Args:
        input_path (str): Path to input audio file.
        vocals_out_dir (str): Directory to save vocals.
        instr_out_dir (str): Directory to save instrumentals.
        trans_out_dir (str): Directory to save transcription (not implemented).
        transcribe (bool): Whether to transcribe vocals (not implemented).
        ai_suffix (str): Suffix for AI tool, e.g. "_S".

    Returns:
        vocals_files (list): List of saved vocal file paths.
        instr_files (list): List of saved instrumental file paths.
        transcription_files (list): List of saved transcription file paths (empty if not implemented).
    """
    # Initialize separator with 2 stems (vocals + accompaniment)
    separator = Separator('spleeter:2stems')

    # Load audio data
    audio_loader = AudioAdapter.default()
    waveform, sample_rate = audio_loader.load(input_path, sample_rate=44100)

    # Perform separation
    prediction = separator.separate(waveform)

    # Prepare output file names
    base_name = os.path.splitext(os.path.basename(input_path))[0]

    # Vocal output path
    vocal_filename = f"{base_name}{ai_suffix}_v.wav"
    vocal_path = os.path.join(vocals_out_dir, vocal_filename)

    # Instrumental output path
    instr_filename = f"{base_name}{ai_suffix}_i.wav"
    instr_path = os.path.join(instr_out_dir, instr_filename)

    # Save separated audio
    audio_loader.save(vocal_path, prediction['vocals'], sample_rate)
    audio_loader.save(instr_path, prediction['accompaniment'], sample_rate)

    vocals_files = [vocal_path]
    instr_files = [instr_path]
    transcription_files = []

    # Transcription not implemented, but placeholder if you want to add later
    if transcribe:
        # You can integrate a transcription tool here and save to trans_out_dir
        # For now, just create an empty file as placeholder
        transcription_filename = f"{base_name}{ai_suffix}_t.txt"
        transcription_path = os.path.join(trans_out_dir, transcription_filename)
        with open(transcription_path, "w") as f:
            f.write("Transcription not implemented yet.\n")
        transcription_files.append(transcription_path)

    return vocals_files, instr_files, transcription_files