"""Attribute extraction utilities for ElevenLabs instrumentation."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterator, Mapping, Optional, Tuple

from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Constants
LLM = OpenInferenceSpanKindValues.LLM.value
CHAIN = OpenInferenceSpanKindValues.CHAIN.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
USER_ID = SpanAttributes.USER_ID

# ElevenLabs specific attribute keys (custom namespace)
ELEVENLABS_VOICE_ID = "elevenlabs.voice_id"
ELEVENLABS_OUTPUT_FORMAT = "elevenlabs.output_format"
ELEVENLABS_OPTIMIZE_STREAMING_LATENCY = "elevenlabs.optimize_streaming_latency"
ELEVENLABS_LANGUAGE_CODE = "elevenlabs.language_code"
ELEVENLABS_CHARACTER_COUNT = "elevenlabs.character_count"
ELEVENLABS_AUDIO_BYTES = "elevenlabs.audio_bytes"
ELEVENLABS_AUDIO_CHUNKS = "elevenlabs.audio_chunks"
ELEVENLABS_AGENT_ID = "elevenlabs.agent_id"
ELEVENLABS_CONVERSATION_ID = "elevenlabs.conversation_id"

# Output format to MIME type mapping
OUTPUT_FORMAT_MIME_TYPES: Dict[Optional[str], str] = {
    "mp3_44100_128": "audio/mpeg",
    "mp3_44100_64": "audio/mpeg",
    "mp3_44100_32": "audio/mpeg",
    "mp3_22050_32": "audio/mpeg",
    "pcm_16000": "audio/pcm",
    "pcm_22050": "audio/pcm",
    "pcm_24000": "audio/pcm",
    "pcm_44100": "audio/pcm",
    "ulaw_8000": "audio/basic",
    None: "audio/mpeg",  # Default
}


def safe_json_dumps(obj: Any) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, default=str)
    except Exception:
        logger.exception("Failed to serialize object to JSON")
        return "{}"


def get_output_mime_type(output_format: Optional[str]) -> str:
    """Get the MIME type for a given output format."""
    return OUTPUT_FORMAT_MIME_TYPES.get(output_format, "audio/mpeg")


def get_llm_span_kind() -> Iterator[Tuple[str, Any]]:
    """Get the span kind attribute for LLM operations."""
    yield OPENINFERENCE_SPAN_KIND, LLM


def get_chain_span_kind() -> Iterator[Tuple[str, Any]]:
    """Get the span kind attribute for chain operations."""
    yield OPENINFERENCE_SPAN_KIND, CHAIN


def get_llm_provider() -> Iterator[Tuple[str, Any]]:
    """Get the LLM provider attribute."""
    yield LLM_PROVIDER, "elevenlabs"


def get_llm_system() -> Iterator[Tuple[str, Any]]:
    """Get the LLM system attribute."""
    yield LLM_SYSTEM, "elevenlabs"


def get_input_value(text: str) -> Iterator[Tuple[str, Any]]:
    """Get the input value attributes."""
    yield INPUT_VALUE, text
    yield INPUT_MIME_TYPE, TEXT


def get_llm_model_name(model_id: Optional[str]) -> Iterator[Tuple[str, Any]]:
    """Get the LLM model name attribute."""
    if model_id:
        yield LLM_MODEL_NAME, model_id


def get_voice_id(voice_id: str) -> Iterator[Tuple[str, Any]]:
    """Get the ElevenLabs voice ID attribute."""
    yield ELEVENLABS_VOICE_ID, voice_id


def get_output_format(output_format: Optional[str]) -> Iterator[Tuple[str, Any]]:
    """Get the output format attribute."""
    if output_format:
        yield ELEVENLABS_OUTPUT_FORMAT, output_format


def get_optimize_streaming_latency(
    optimize_streaming_latency: Optional[int],
) -> Iterator[Tuple[str, Any]]:
    """Get the optimize streaming latency attribute."""
    if optimize_streaming_latency is not None:
        yield ELEVENLABS_OPTIMIZE_STREAMING_LATENCY, optimize_streaming_latency


def get_language_code(language_code: Optional[str]) -> Iterator[Tuple[str, Any]]:
    """Get the language code attribute."""
    if language_code:
        yield ELEVENLABS_LANGUAGE_CODE, language_code


def get_character_count(text: str) -> Iterator[Tuple[str, Any]]:
    """Get the character count attribute."""
    yield ELEVENLABS_CHARACTER_COUNT, len(text)


def get_output_value(output_format: Optional[str], byte_count: int) -> Iterator[Tuple[str, Any]]:
    """Get the output value attributes for audio response."""
    mime_type = get_output_mime_type(output_format)
    yield OUTPUT_VALUE, f"{mime_type}, {byte_count} bytes"
    yield OUTPUT_MIME_TYPE, mime_type


def get_audio_bytes(byte_count: int) -> Iterator[Tuple[str, Any]]:
    """Get the audio bytes attribute."""
    yield ELEVENLABS_AUDIO_BYTES, byte_count


def get_audio_chunks(chunk_count: int) -> Iterator[Tuple[str, Any]]:
    """Get the audio chunks attribute."""
    yield ELEVENLABS_AUDIO_CHUNKS, chunk_count


def get_invocation_parameters(kwargs: Mapping[str, Any]) -> Iterator[Tuple[str, Any]]:
    """Extract and serialize invocation parameters."""
    params: Dict[str, Any] = {}

    # Voice settings (can be a VoiceSettings object or dict)
    voice_settings = kwargs.get("voice_settings")
    if voice_settings is not None:
        # Handle both dict and VoiceSettings object
        if hasattr(voice_settings, "model_dump"):
            params["voice_settings"] = voice_settings.model_dump()
        elif hasattr(voice_settings, "__dict__"):
            params["voice_settings"] = {
                k: v for k, v in voice_settings.__dict__.items() if not k.startswith("_")
            }
        elif isinstance(voice_settings, dict):
            params["voice_settings"] = voice_settings

    # Other relevant parameters
    param_keys = [
        "optimize_streaming_latency",
        "output_format",
        "language_code",
        "apply_text_normalization",
        "seed",
    ]
    for key in param_keys:
        value = kwargs.get(key)
        if value is not None:
            params[key] = value

    if params:
        yield LLM_INVOCATION_PARAMETERS, safe_json_dumps(params)


def get_tts_request_attributes(
    voice_id: str,
    text: str,
    kwargs: Mapping[str, Any],
) -> Iterator[Tuple[str, Any]]:
    """Get all request attributes for a TTS call."""
    # Span kind and provider
    yield from get_llm_span_kind()
    yield from get_llm_provider()
    yield from get_llm_system()

    # Input
    yield from get_input_value(text)

    # Model
    model_id = kwargs.get("model_id")
    yield from get_llm_model_name(model_id)

    # ElevenLabs specific
    yield from get_voice_id(voice_id)
    yield from get_output_format(kwargs.get("output_format"))
    yield from get_optimize_streaming_latency(kwargs.get("optimize_streaming_latency"))
    yield from get_language_code(kwargs.get("language_code"))

    # Invocation parameters
    yield from get_invocation_parameters(kwargs)


def get_tts_response_attributes(
    text: str,
    output_format: Optional[str],
    byte_count: int,
    chunk_count: int = 0,
) -> Iterator[Tuple[str, Any]]:
    """Get all response attributes for a TTS call."""
    yield from get_character_count(text)
    yield from get_output_value(output_format, byte_count)
    yield from get_audio_bytes(byte_count)
    if chunk_count > 0:
        yield from get_audio_chunks(chunk_count)


def get_conversation_attributes(
    agent_id: Optional[str],
    user_id: Optional[str],
) -> Iterator[Tuple[str, Any]]:
    """Get attributes for a conversation session."""
    yield from get_chain_span_kind()
    yield from get_llm_provider()
    yield from get_llm_system()

    if agent_id:
        yield ELEVENLABS_AGENT_ID, agent_id
    if user_id:
        yield USER_ID, user_id


def get_conversation_end_attributes(
    conversation_id: Optional[str],
) -> Iterator[Tuple[str, Any]]:
    """Get attributes for conversation end."""
    if conversation_id:
        yield ELEVENLABS_CONVERSATION_ID, conversation_id
