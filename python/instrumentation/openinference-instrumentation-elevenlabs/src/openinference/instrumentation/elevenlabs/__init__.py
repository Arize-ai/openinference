"""OpenInference instrumentation for ElevenLabs."""

import logging
from typing import Any, Collection

from opentelemetry import trace as trace_api
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor  # type: ignore
from wrapt import wrap_function_wrapper

from openinference.instrumentation import OITracer, TraceConfig

from ._conversation import (
    _ConversationEndSessionWrapper,
    _ConversationStartSessionWrapper,
    _ConversationWaitForSessionEndWrapper,
)
from ._wrappers import (
    _AsyncTextToSpeechConvertWithTimestampsWrapper,
    _AsyncTextToSpeechConvertWrapper,
    _AsyncTextToSpeechStreamWithTimestampsWrapper,
    _AsyncTextToSpeechStreamWrapper,
    _TextToSpeechConvertWithTimestampsWrapper,
    _TextToSpeechConvertWrapper,
    _TextToSpeechStreamWithTimestampsWrapper,
    _TextToSpeechStreamWrapper,
)
from .package import _instruments
from .version import __version__

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

__all__ = ["ElevenLabsInstrumentor"]


class ElevenLabsInstrumentor(BaseInstrumentor):  # type: ignore
    """
    An instrumentor for ElevenLabs.

    Automatically instruments ElevenLabs API calls to create
    OpenInference-compliant spans for:
    - Text-to-Speech operations (convert, stream, with timestamps)
    - Conversational AI sessions
    """

    __slots__ = (
        "_tracer",
        "_config",
        # TTS sync methods
        "_original_tts_convert",
        "_original_tts_convert_with_timestamps",
        "_original_tts_stream",
        "_original_tts_stream_with_timestamps",
        # TTS async methods
        "_original_async_tts_convert",
        "_original_async_tts_convert_with_timestamps",
        "_original_async_tts_stream",
        "_original_async_tts_stream_with_timestamps",
        # Conversation methods
        "_original_conversation_start_session",
        "_original_conversation_end_session",
        "_original_conversation_wait_for_session_end",
    )

    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments if isinstance(_instruments, tuple) else ()

    def _instrument(self, **kwargs: Any) -> None:
        """
        Instrument ElevenLabs.

        Args:
            tracer_provider: OpenTelemetry TracerProvider
            config: OpenInference TraceConfig
        """
        if not (tracer_provider := kwargs.get("tracer_provider")):
            tracer_provider = trace_api.get_tracer_provider()

        if not (config := kwargs.get("config")):
            config = TraceConfig()
        else:
            assert isinstance(config, TraceConfig)

        # Create OITracer
        tracer = OITracer(
            trace_api.get_tracer(__name__, __version__, tracer_provider),
            config=config,
        )

        # Store for later use
        self._tracer = tracer
        self._config = config

        # Import ElevenLabs modules and wrap methods
        self._instrument_text_to_speech(tracer)
        self._instrument_conversation(tracer)

        logger.info("ElevenLabs instrumentation enabled")

    def _instrument_text_to_speech(self, tracer: OITracer) -> None:
        """Instrument text-to-speech methods."""
        try:
            from elevenlabs.text_to_speech.client import (
                AsyncTextToSpeechClient,
                TextToSpeechClient,
            )

            # Store originals for uninstrument
            self._original_tts_convert = TextToSpeechClient.convert
            self._original_tts_convert_with_timestamps = TextToSpeechClient.convert_with_timestamps
            self._original_tts_stream = TextToSpeechClient.stream
            self._original_tts_stream_with_timestamps = TextToSpeechClient.stream_with_timestamps

            self._original_async_tts_convert = AsyncTextToSpeechClient.convert
            self._original_async_tts_convert_with_timestamps = (
                AsyncTextToSpeechClient.convert_with_timestamps
            )
            self._original_async_tts_stream = AsyncTextToSpeechClient.stream
            self._original_async_tts_stream_with_timestamps = (
                AsyncTextToSpeechClient.stream_with_timestamps
            )

            # Wrap sync methods
            wrap_function_wrapper(
                module="elevenlabs.text_to_speech.client",
                name="TextToSpeechClient.convert",
                wrapper=_TextToSpeechConvertWrapper(tracer=tracer),
            )
            wrap_function_wrapper(
                module="elevenlabs.text_to_speech.client",
                name="TextToSpeechClient.convert_with_timestamps",
                wrapper=_TextToSpeechConvertWithTimestampsWrapper(tracer=tracer),
            )
            wrap_function_wrapper(
                module="elevenlabs.text_to_speech.client",
                name="TextToSpeechClient.stream",
                wrapper=_TextToSpeechStreamWrapper(tracer=tracer),
            )
            wrap_function_wrapper(
                module="elevenlabs.text_to_speech.client",
                name="TextToSpeechClient.stream_with_timestamps",
                wrapper=_TextToSpeechStreamWithTimestampsWrapper(tracer=tracer),
            )

            # Wrap async methods
            wrap_function_wrapper(
                module="elevenlabs.text_to_speech.client",
                name="AsyncTextToSpeechClient.convert",
                wrapper=_AsyncTextToSpeechConvertWrapper(tracer=tracer),
            )
            wrap_function_wrapper(
                module="elevenlabs.text_to_speech.client",
                name="AsyncTextToSpeechClient.convert_with_timestamps",
                wrapper=_AsyncTextToSpeechConvertWithTimestampsWrapper(tracer=tracer),
            )
            wrap_function_wrapper(
                module="elevenlabs.text_to_speech.client",
                name="AsyncTextToSpeechClient.stream",
                wrapper=_AsyncTextToSpeechStreamWrapper(tracer=tracer),
            )
            wrap_function_wrapper(
                module="elevenlabs.text_to_speech.client",
                name="AsyncTextToSpeechClient.stream_with_timestamps",
                wrapper=_AsyncTextToSpeechStreamWithTimestampsWrapper(tracer=tracer),
            )

            logger.debug("Text-to-speech instrumentation complete")

        except ImportError:
            logger.warning(
                "Could not import elevenlabs.text_to_speech.client, TTS instrumentation skipped"
            )
        except Exception:
            logger.exception("Failed to instrument text-to-speech")

    def _instrument_conversation(self, tracer: OITracer) -> None:
        """Instrument conversation methods."""
        try:
            from elevenlabs.conversational_ai.conversation import Conversation

            # Store originals for uninstrument
            # Note: We don't wrap __init__ as wrapt doesn't support it well
            self._original_conversation_start_session = Conversation.start_session
            self._original_conversation_end_session = Conversation.end_session
            self._original_conversation_wait_for_session_end = Conversation.wait_for_session_end

            # Wrap conversation methods (excluding __init__)
            # The start_session wrapper will create the span
            wrap_function_wrapper(
                module="elevenlabs.conversational_ai.conversation",
                name="Conversation.start_session",
                wrapper=_ConversationStartSessionWrapper(tracer=tracer),
            )
            wrap_function_wrapper(
                module="elevenlabs.conversational_ai.conversation",
                name="Conversation.end_session",
                wrapper=_ConversationEndSessionWrapper(tracer=tracer),
            )
            wrap_function_wrapper(
                module="elevenlabs.conversational_ai.conversation",
                name="Conversation.wait_for_session_end",
                wrapper=_ConversationWaitForSessionEndWrapper(tracer=tracer),
            )

            logger.debug("Conversation instrumentation complete")

        except ImportError:
            logger.warning(
                "Could not import elevenlabs.conversational_ai.conversation, "
                "conversation instrumentation skipped"
            )
        except Exception:
            logger.exception("Failed to instrument conversation")

    def _uninstrument(self, **kwargs: Any) -> None:
        """
        Uninstrument ElevenLabs.
        """
        # Restore TTS sync methods
        try:
            from elevenlabs.text_to_speech.client import TextToSpeechClient

            if hasattr(self, "_original_tts_convert"):
                TextToSpeechClient.convert = self._original_tts_convert  # type: ignore
            if hasattr(self, "_original_tts_convert_with_timestamps"):
                TextToSpeechClient.convert_with_timestamps = (  # type: ignore
                    self._original_tts_convert_with_timestamps
                )
            if hasattr(self, "_original_tts_stream"):
                TextToSpeechClient.stream = self._original_tts_stream  # type: ignore
            if hasattr(self, "_original_tts_stream_with_timestamps"):
                TextToSpeechClient.stream_with_timestamps = (  # type: ignore
                    self._original_tts_stream_with_timestamps
                )
        except ImportError:
            pass

        # Restore TTS async methods
        try:
            from elevenlabs.text_to_speech.client import AsyncTextToSpeechClient

            if hasattr(self, "_original_async_tts_convert"):
                AsyncTextToSpeechClient.convert = (  # type: ignore
                    self._original_async_tts_convert
                )
            if hasattr(self, "_original_async_tts_convert_with_timestamps"):
                AsyncTextToSpeechClient.convert_with_timestamps = (  # type: ignore
                    self._original_async_tts_convert_with_timestamps
                )
            if hasattr(self, "_original_async_tts_stream"):
                AsyncTextToSpeechClient.stream = (  # type: ignore
                    self._original_async_tts_stream
                )
            if hasattr(self, "_original_async_tts_stream_with_timestamps"):
                AsyncTextToSpeechClient.stream_with_timestamps = (  # type: ignore
                    self._original_async_tts_stream_with_timestamps
                )
        except ImportError:
            pass

        # Restore conversation methods
        try:
            from elevenlabs.conversational_ai.conversation import Conversation

            if hasattr(self, "_original_conversation_start_session"):
                Conversation.start_session = (  # type: ignore
                    self._original_conversation_start_session
                )
            if hasattr(self, "_original_conversation_end_session"):
                Conversation.end_session = (  # type: ignore
                    self._original_conversation_end_session
                )
            if hasattr(self, "_original_conversation_wait_for_session_end"):
                Conversation.wait_for_session_end = (  # type: ignore
                    self._original_conversation_wait_for_session_end
                )
        except ImportError:
            pass

        logger.info("ElevenLabs instrumentation disabled")
