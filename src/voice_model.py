"""
Whisper-based speech-to-text for abuse detection pipeline.
Loads once at startup; uses GPU when available.
"""

import os
import io
import tempfile
import torch
import whisper
import numpy as np

_whisper_model = None


def load_whisper(model_size: str = "small") -> whisper.Whisper:
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Whisper '{model_size}' on {device.upper()}...")
    _whisper_model = whisper.load_model(model_size, device=device)
    print(f"  Whisper ready.")
    return _whisper_model


def transcribe(audio_bytes: bytes, ext: str = ".webm") -> str:
    """
    Transcribe raw audio bytes.  ext is the file extension ('.webm', '.wav', etc.)
    Returns the transcribed string, or raises RuntimeError on failure.
    """
    model = load_whisper()

    # Write to a temp file so ffmpeg can decode it
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = model.transcribe(
            tmp_path,
            language="en",
            fp16=torch.cuda.is_available(),
            temperature=0.0,
        )
        text = result["text"].strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not text:
        raise RuntimeError("No speech detected in the recording.")

    return text
