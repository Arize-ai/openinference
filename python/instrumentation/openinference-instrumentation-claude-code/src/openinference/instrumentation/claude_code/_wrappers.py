"""Wrappers for Claude Code SDK methods."""

import logging
from typing import Any, AsyncIterator, Callable, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import Span

from openinference.instrumentation.claude_code._message_parser import extract_text_content
from openinference.instrumentation.claude_code._span_manager import SpanManager

logger = logging.getLogger(__name__)


class _QueryWrapper:
    """Wrapper for claude_agent_sdk.query() function."""

    __slots__ = ("_tracer", "_span_manager")

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer
        self._span_manager = SpanManager(tracer)

    async def __call__(
        self,
        wrapped: Callable[..., AsyncIterator[Any]],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Wrap query() function."""
        # Check suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for message in wrapped(*args, **kwargs):
                yield message
            return

        # Extract prompt from kwargs
        prompt = kwargs.get("prompt", "")
        session_id = f"query-{id(prompt)}"  # Generate session ID

        # Start root AGENT span
        root_span = self._span_manager.start_agent_span(
            name="Claude Code Query Session",
            session_id=session_id,
        )

        try:
            # Start LLM span for query
            query_span = self._span_manager.start_llm_span(
                name="Claude Code Query",
                parent_span=root_span,
            )

            try:
                # Call original function and yield messages
                async for message in wrapped(*args, **kwargs):
                    # TODO: Parse message and create child spans
                    yield message
            finally:
                self._span_manager.end_span(query_span)
        finally:
            self._span_manager.end_span(root_span)
