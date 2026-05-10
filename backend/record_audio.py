# backend/record_audio.py

import os
import shutil
import subprocess
import uuid
from fastapi import UploadFile
from typing import Tuple

SAMPLE_RATE = 16000

def record_audio(file: UploadFile) -> Tuple[str, str]:
    """
    Save and convert uploaded audio file to WAV format using unique temporary filenames.

    Args:
        file: UploadFile from FastAPI

    Returns:
        Tuple of (wav_path, webm_path) - both temporary files that need cleanup
    """
    print("🎙 Receiving audio from frontend...")

    # Create unique temporary filenames to avoid clobbering under concurrent requests
    unique_id = uuid.uuid4().hex
    temp_webm = f"temp_audio_{unique_id}.webm"
    temp_wav = f"raw_audio_{unique_id}.wav"

    # Save raw uploaded file
    with open(temp_webm, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert to WAV using ffmpeg
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", temp_webm,
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                temp_wav
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True
        )
    except Exception:
        # Clean up temp files on conversion failure
        if os.path.exists(temp_webm):
            os.remove(temp_webm)
        if os.path.exists(temp_wav):
            os.remove(temp_wav)
        raise

    print(f"✅ Recording saved as {temp_wav}")

    # Return both paths for cleanup
    return (temp_wav, temp_webm)
