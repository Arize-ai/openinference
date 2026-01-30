"""Wrapper classes for ElevenLabs TTS method instrumentation."""

from __future__ import annotations

import logging
from abc import ABC
from contextlib import contextmanager
from itertools import chain
from typing import Any, Callable, Iterator, Mapping, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN

from openinference.instrumentation import get_attributes_from_context

from ._attributes import get_tts_request_attributes, get_tts_response_attributes
from ._stream import (
    _AsyncAudioStream,
    _AsyncTimestampedAudioStream,
    _AudioStream,
    _TimestampedAudioStream,
)
from ._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _WithTracer(ABC):
    """Base class for wrappers that need a tracer."""

    def __init__(self, tracer: trace_api.Tracer, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracer = tracer

    @contextmanager
    def _start_as_current_span(
        self, span_name: str, attributes: Mapping[str, Any]
    ) -> Iterator[_WithSpan]:
        """Start a span and yield a _WithSpan helper."""
        try:
            span = self._tracer.start_span(
                name=span_name,
                record_exception=False,
                set_status_on_exception=False,
                attributes=dict(attributes),
            )
        except Exception:
            logger.exception("Failed to start span")
            span = INVALID_SPAN

        with trace_api.use_span(
            span,
            end_on_exit=False,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            yield _WithSpan(span=span)


class _TextToSpeechConvertWrapper(_WithTracer):
    """Wrapper for sync text_to_speech.convert() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Check for suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Extract parameters - voice_id is first positional arg
        voice_id = args[0] if args else kwargs.get("voice_id", "")
        text = kwargs.get("text", "")
        output_format = kwargs.get("output_format")

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_tts_request_attributes(voice_id, text, kwargs),
            )
        )

        with self._start_as_current_span(
            span_name="ElevenLabs.TextToSpeech",
            attributes=attributes,
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise

            # convert() returns an Iterator[bytes], wrap it to track completion
            return _AudioStream(
                stream=response,
                with_span=span,
                text=text,
                output_format=output_format,
            )


class _AsyncTextToSpeechConvertWrapper(_WithTracer):
    """Wrapper for async text_to_speech.convert() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Check for suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Extract parameters
        voice_id = args[0] if args else kwargs.get("voice_id", "")
        text = kwargs.get("text", "")
        output_format = kwargs.get("output_format")

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_tts_request_attributes(voice_id, text, kwargs),
            )
        )

        with self._start_as_current_span(
            span_name="ElevenLabs.TextToSpeech",
            attributes=attributes,
        ) as span:
            try:
                # Async convert() returns an AsyncIterator[bytes] directly (not awaitable)
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise

            return _AsyncAudioStream(
                stream=response,
                with_span=span,
                text=text,
                output_format=output_format,
            )


class _TextToSpeechConvertWithTimestampsWrapper(_WithTracer):
    """Wrapper for sync text_to_speech.convert_with_timestamps() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Check for suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Extract parameters
        voice_id = args[0] if args else kwargs.get("voice_id", "")
        text = kwargs.get("text", "")
        output_format = kwargs.get("output_format")

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_tts_request_attributes(voice_id, text, kwargs),
            )
        )

        with self._start_as_current_span(
            span_name="ElevenLabs.TextToSpeech",
            attributes=attributes,
        ) as span:
            try:
                # convert_with_timestamps returns a single AudioWithTimestampsResponse
                response = wrapped(*args, **kwargs)

                # Response has audio_base_64 attribute - estimate byte count
                byte_count = 0
                audio_data = getattr(response, "audio_base_64", None)
                if audio_data:
                    # Base64 encoded, actual bytes are ~75% of string length
                    byte_count = int(len(audio_data) * 0.75)
                else:
                    logger.debug(
                        f"convert_with_timestamps response has no audio_base_64: "
                        f"type={type(response)}, attrs={dir(response)}"
                    )

                # Set response attributes and finish
                response_attrs = dict(get_tts_response_attributes(text, output_format, byte_count))
                span.set_status(trace_api.Status(trace_api.StatusCode.OK))
                span.finish_tracing(attributes=response_attrs)

                return response
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise


class _AsyncTextToSpeechConvertWithTimestampsWrapper(_WithTracer):
    """Wrapper for async text_to_speech.convert_with_timestamps() method."""

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Check for suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)

        # Extract parameters
        voice_id = args[0] if args else kwargs.get("voice_id", "")
        text = kwargs.get("text", "")
        output_format = kwargs.get("output_format")

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_tts_request_attributes(voice_id, text, kwargs),
            )
        )

        with self._start_as_current_span(
            span_name="ElevenLabs.TextToSpeech",
            attributes=attributes,
        ) as span:
            try:
                # Async convert_with_timestamps returns a coroutine -> single response
                response = await wrapped(*args, **kwargs)

                # Response has audio_base_64 attribute - estimate byte count
                byte_count = 0
                audio_data = getattr(response, "audio_base_64", None)
                if audio_data:
                    # Base64 encoded, actual bytes are ~75% of string length
                    byte_count = int(len(audio_data) * 0.75)
                else:
                    logger.debug(
                        f"async convert_with_timestamps response has no audio_base_64: "
                        f"type={type(response)}, attrs={dir(response)}"
                    )

                # Set response attributes and finish
                response_attrs = dict(get_tts_response_attributes(text, output_format, byte_count))
                span.set_status(trace_api.Status(trace_api.StatusCode.OK))
                span.finish_tracing(attributes=response_attrs)

                return response
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise


class _TextToSpeechStreamWrapper(_WithTracer):
    """Wrapper for sync text_to_speech.stream() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Check for suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Extract parameters
        voice_id = args[0] if args else kwargs.get("voice_id", "")
        text = kwargs.get("text", "")
        output_format = kwargs.get("output_format")

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_tts_request_attributes(voice_id, text, kwargs),
            )
        )

        with self._start_as_current_span(
            span_name="ElevenLabs.TextToSpeech",
            attributes=attributes,
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise

            # stream() returns Iterator[bytes], wrap it
            return _AudioStream(
                stream=response,
                with_span=span,
                text=text,
                output_format=output_format,
            )


class _AsyncTextToSpeechStreamWrapper(_WithTracer):
    """Wrapper for async text_to_speech.stream() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Check for suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Extract parameters
        voice_id = args[0] if args else kwargs.get("voice_id", "")
        text = kwargs.get("text", "")
        output_format = kwargs.get("output_format")

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_tts_request_attributes(voice_id, text, kwargs),
            )
        )

        with self._start_as_current_span(
            span_name="ElevenLabs.TextToSpeech",
            attributes=attributes,
        ) as span:
            try:
                # Async stream() returns AsyncIterator[bytes] directly (not awaitable)
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise

            return _AsyncAudioStream(
                stream=response,
                with_span=span,
                text=text,
                output_format=output_format,
            )


class _TextToSpeechStreamWithTimestampsWrapper(_WithTracer):
    """Wrapper for sync text_to_speech.stream_with_timestamps() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Check for suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Extract parameters
        voice_id = args[0] if args else kwargs.get("voice_id", "")
        text = kwargs.get("text", "")
        output_format = kwargs.get("output_format")

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_tts_request_attributes(voice_id, text, kwargs),
            )
        )

        with self._start_as_current_span(
            span_name="ElevenLabs.TextToSpeech",
            attributes=attributes,
        ) as span:
            try:
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise

            # stream_with_timestamps returns Iterator[StreamingAudioChunkWithTimestampsResponse]
            return _TimestampedAudioStream(
                stream=response,
                with_span=span,
                text=text,
                output_format=output_format,
            )


class _AsyncTextToSpeechStreamWithTimestampsWrapper(_WithTracer):
    """Wrapper for async text_to_speech.stream_with_timestamps() method."""

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        # Check for suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)

        # Extract parameters
        voice_id = args[0] if args else kwargs.get("voice_id", "")
        text = kwargs.get("text", "")
        output_format = kwargs.get("output_format")

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_tts_request_attributes(voice_id, text, kwargs),
            )
        )

        with self._start_as_current_span(
            span_name="ElevenLabs.TextToSpeech",
            attributes=attributes,
        ) as span:
            try:
                # Async stream_with_timestamps returns AsyncIterator directly (not awaitable)
                response = wrapped(*args, **kwargs)
            except Exception as exception:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, str(exception)))
                span.record_exception(exception)
                span.finish_tracing()
                raise

            return _AsyncTimestampedAudioStream(
                stream=response,
                with_span=span,
                text=text,
                output_format=output_format,
            )
