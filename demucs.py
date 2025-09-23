import os
import shlex
import demucs.separate

def separate_D(input_path, vocals_out_dir, instr_out_dir, trans_out_dir, transcribe, ai_suffix="_D"):
    """
    Separate audio using Demucs.

    Args:
        input_path (str): Path to input audio file.
        vocals_out_dir (str): Directory to save vocals.
        instr_out_dir (str): Directory to save instrumentals.
        trans_out_dir (str): Directory to save transcription (not implemented).
        transcribe (bool): Whether to transcribe vocals (not implemented).
        ai_suffix (str): Suffix for AI tool, e.g. "_D".

    Returns:
        vocals_files (list): List of saved vocal file paths.
        instr_files (list): List of saved instrumental file paths.
        transcription_files (list): List of saved transcription file paths (empty if not implemented).
    """
    # Prepare output base folder for Demucs (Demucs creates subfolders automatically)
    # We'll use a temporary output folder and then move files to desired folders with suffixes

    # Create a temporary output folder inside instr_out_dir (or use a temp folder)
    temp_output_dir = os.path.join(instr_out_dir, "demucs_temp_output")
    os.makedirs(temp_output_dir, exist_ok=True)

    # Build Demucs command arguments
    # --mp3 to save as mp3, --two-stems vocals to separate vocals and accompaniment only
    # -n mdx_extra is a good model, you can change if you want
    args = [
        "--mp3",
        "--two-stems", "vocals",
        "-n", "mdx_extra",
        "-o", temp_output_dir,
        input_path
    ]

    # Run Demucs separation
    demucs.separate.main(args)

    # After separation, Demucs creates a folder inside temp_output_dir with the input file base name
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    demucs_output_folder = os.path.join(temp_output_dir, base_name)

    vocals_files = []
    instr_files = []
    transcription_files = []

    if not os.path.isdir(demucs_output_folder):
        raise FileNotFoundError(f"Demucs output folder not found: {demucs_output_folder}")

    # Demucs outputs two files: vocals.mp3 and accompaniment.mp3 inside demucs_output_folder
    vocals_src = os.path.join(demucs_output_folder, "vocals.mp3")
    instr_src = os.path.join(demucs_output_folder, "accompaniment.mp3")

    # Prepare destination paths with suffixes
    vocal_filename = f"{base_name}{ai_suffix}_v.mp3"
    instr_filename = f"{base_name}{ai_suffix}_i.mp3"

    vocal_dest = os.path.join(vocals_out_dir, vocal_filename)
    instr_dest = os.path.join(instr_out_dir, instr_filename)

    # Move or copy files to final output folders
    os.replace(vocals_src, vocal_dest)
    os.replace(instr_src, instr_dest)

    vocals_files.append(vocal_dest)
    instr_files.append(instr_dest)

    # Clean up temp output folder
    try:
        import shutil
        shutil.rmtree(temp_output_dir)
    except Exception:
        pass

    # Transcription not implemented, placeholder if needed
    if transcribe:
        transcription_filename = f"{base_name}{ai_suffix}_t.txt"
        transcription_path = os.path.join(trans_out_dir, transcription_filename)
        with open(transcription_path, "w") as f:
            f.write("Transcription not implemented yet.\n")
        transcription_files.append(transcription_path)

    return vocals_files, instr_files, transcription_files