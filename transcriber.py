"""
transcriber.py

Transcribes audio or video files to text using OpenAI's open-source
Whisper model, running fully locally (no API key, no paid service).

Whisper uses ffmpeg internally to decode the input, so this works
directly on video files (mp4, mov, mkv, webm, ...) as well as plain
audio files (mp3, wav, m4a, ...) -- you do not need to separately
extract the audio track first.

Requires the `ffmpeg` command line tool to be installed on your
system (see README.md).
"""

import whisper

_MODEL_CACHE = {}


def load_whisper_model(model_size: str = "base"):
    """Load (and cache) a Whisper model of the given size."""
    if model_size not in _MODEL_CACHE:
        _MODEL_CACHE[model_size] = whisper.load_model(model_size)
    return _MODEL_CACHE[model_size]


def transcribe_media(file_path: str, model_size: str = "base", language: str = None) -> str:
    """
    Transcribe the audio/video file at `file_path` to plain text.
    `model_size` is one of: tiny, base, small, medium, large.
    `language` can be left as None for auto-detection.
    """
    model = load_whisper_model(model_size)
    options = {}
    if language:
        options["language"] = language
    result = model.transcribe(file_path, **options)
    return (result.get("text") or "").strip()
