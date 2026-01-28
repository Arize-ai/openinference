"""Conversation class instrumentation for ElevenLabs."""

from __future__ import annotations

import logging
from itertools import chain
from typing import Any, Callable, Mapping, Optional, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import INVALID_SPAN

from openinference.instrumentation import get_attributes_from_context

from ._attributes import get_conversation_attributes, get_conversation_end_attributes
from ._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Storage key for the span on Conversation instances
_CONVERSATION_SPAN_KEY = "_openinference_span"
_CONVERSATION_WITH_SPAN_KEY = "_openinference_with_span"


class _ConversationStartSessionWrapper:
    """Wrapper for Conversation.start_session() to create a session span."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

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

        # Extract parameters from the instance (set during __init__)
        agent_id = getattr(instance, "agent_id", None)
        user_id = getattr(instance, "user_id", None)

        # Build attributes
        attributes = dict(
            chain(
                get_attributes_from_context(),
                get_conversation_attributes(agent_id, user_id),
            )
        )

        # Start the session span
        try:
            span = self._tracer.start_span(
                name="ElevenLabs.Conversation",
                record_exception=False,
                set_status_on_exception=False,
                attributes=attributes,
            )
        except Exception:
            logger.exception("Failed to start conversation span")
            span = INVALID_SPAN

        # Create the WithSpan helper
        with_span = _WithSpan(span=span)

        # Store the span and helper on the instance for later access
        try:
            setattr(instance, _CONVERSATION_SPAN_KEY, span)
            setattr(instance, _CONVERSATION_WITH_SPAN_KEY, with_span)
        except Exception:
            logger.exception("Failed to store span on Conversation instance")

        # Add session_started event
        if span.is_recording():
            try:
                span.add_event("session_started")
            except Exception:
                logger.exception("Failed to add session_started event")

        return wrapped(*args, **kwargs)


class _ConversationEndSessionWrapper:
    """Wrapper for Conversation.end_session()."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

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

        # Call the original method first
        result = wrapped(*args, **kwargs)

        # End the span
        with_span: Optional[_WithSpan] = getattr(instance, _CONVERSATION_WITH_SPAN_KEY, None)

        if with_span is not None and not with_span.is_finished:
            with_span.finish_tracing(
                status=trace_api.Status(trace_api.StatusCode.OK),
            )

        return result


class _ConversationWaitForSessionEndWrapper:
    """Wrapper for Conversation.wait_for_session_end()."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

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

        # Call the original method - this blocks until session ends
        # and returns the conversation_id
        result = wrapped(*args, **kwargs)

        # End the span with the conversation_id
        with_span: Optional[_WithSpan] = getattr(instance, _CONVERSATION_WITH_SPAN_KEY, None)

        if with_span is not None and not with_span.is_finished:
            # result is the conversation_id
            end_attrs = dict(get_conversation_end_attributes(result))

            with_span.finish_tracing(
                status=trace_api.Status(trace_api.StatusCode.OK),
                attributes=end_attrs,
            )

        return result
