"""
Helpers for normalizing binary media (audio, files) found in OpenAI
requests and responses into base64 data URIs.

OpenInference records inline media as ``data:<mime>;base64,<payload>`` URIs
on ``audio.url`` / ``file.url`` attributes so that a single, central
``TraceConfig`` rule can redact, truncate, or externalize them (see the
blob upload support in ``openinference-instrumentation``).
"""

import mimetypes
from typing import Optional

__all__ = (
    "get_audio_mime_type",
    "get_audio_data_uri",
    "get_file_data_uri",
    "guess_file_mime_type",
)

# Mime types for the audio formats accepted/produced by the OpenAI API.
_AUDIO_MIME_BY_FORMAT = {
    "aac": "audio/aac",
    "flac": "audio/flac",
    "g711_alaw": "audio/g711-alaw",
    "g711_ulaw": "audio/g711-ulaw",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "pcm": "audio/pcm",
    "pcm16": "audio/pcm",
    "wav": "audio/wav",
}

_DEFAULT_AUDIO_MIME_TYPE = "audio/wav"
_DEFAULT_FILE_MIME_TYPE = "application/octet-stream"


def get_audio_mime_type(audio_format: Optional[str]) -> str:
    if not audio_format:
        return _DEFAULT_AUDIO_MIME_TYPE
    return _AUDIO_MIME_BY_FORMAT.get(audio_format.lower(), f"audio/{audio_format.lower()}")


def get_audio_data_uri(base64_data: str, audio_format: Optional[str]) -> str:
    return f"data:{get_audio_mime_type(audio_format)};base64,{base64_data}"


def guess_file_mime_type(filename: Optional[str]) -> str:
    if filename:
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type
    return _DEFAULT_FILE_MIME_TYPE


def get_file_data_uri(file_data: str, filename: Optional[str]) -> str:
    """
    Normalizes OpenAI ``file_data`` (which may already be a data URI or a
    bare base64 string) into a data URI.
    """
    if file_data.startswith("data:"):
        return file_data
    return f"data:{guess_file_mime_type(filename)};base64,{file_data}"
