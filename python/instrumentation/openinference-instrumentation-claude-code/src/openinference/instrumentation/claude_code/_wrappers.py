"""Wrappers for Claude Code SDK methods."""

import logging
from typing import Any, AsyncIterator, Callable, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import Span

from openinference.instrumentation.claude_code._message_parser import (
    extract_text_content,
    extract_tool_uses,
)
from openinference.instrumentation.claude_code._span_manager import SpanManager
from openinference.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)


class _ClientQueryWrapper:
    """Wrapper for ClaudeSDKClient.query() method to capture input."""

    __slots__ = ("_tracer",)

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Any,
    ) -> Any:
        """Wrap query() method to store input on client instance."""
        # Store the query for later use in receive_response
        if args:
            instance._otel_last_query = args[0]  # First arg is the query string

        # Call the original method
        return await wrapped(*args, **kwargs)


class _ReceiveResponseWrapper:
    """Wrapper for ClaudeSDKClient.receive_response() method."""

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
        """Wrap receive_response() method."""
        # Check suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for message in wrapped(*args, **kwargs):
                yield message
            return

        # Generate session ID from client instance
        session_id = f"client-{id(instance)}"

        # Start root AGENT span
        root_span = self._span_manager.start_agent_span(
            name="Claude Code Client Session",
            session_id=session_id,
        )

        try:
            # Start LLM span for response
            response_span = self._span_manager.start_llm_span(
                name="Claude Code Client Response",
                parent_span=root_span,
            )

            # Set input message if available
            if hasattr(instance, "_otel_last_query"):
                response_span.set_attribute(
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role",
                    "user",
                )
                response_span.set_attribute(
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.content",
                    instance._otel_last_query,
                )

            try:
                # Track tool spans by ID and collect output
                tool_spans = {}
                output_messages = []

                # Call original function and yield messages
                async for message in wrapped(*args, **kwargs):
                    # Parse message for tool uses
                    from claude_agent_sdk import AssistantMessage

                    if isinstance(message, AssistantMessage):
                        # Collect output for span attributes
                        output_messages.append(message)

                        # Extract and create tool spans
                        tool_uses = extract_tool_uses(message)
                        for tool_use in tool_uses:
                            tool_span = self._span_manager.start_tool_span(
                                tool_name=tool_use["name"],
                                parent_span=response_span,
                            )
                            # Set tool parameters
                            tool_span.set_attribute(
                                SpanAttributes.TOOL_PARAMETERS,
                                str(tool_use["input"]),
                            )
                            # Track for later closing
                            tool_spans[tool_use["id"]] = tool_span

                    yield message

                # Set LLM output messages on span
                if output_messages:
                    # Combine all text content from messages
                    output_text = "\n".join(
                        extract_text_content(msg) for msg in output_messages
                    )
                    if output_text:
                        response_span.set_attribute(
                            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.role",
                            "assistant",
                        )
                        response_span.set_attribute(
                            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.content",
                            output_text,
                        )

                # End all tool spans
                for tool_span in tool_spans.values():
                    self._span_manager.end_span(tool_span)

            finally:
                self._span_manager.end_span(response_span)
        finally:
            self._span_manager.end_span(root_span)


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
                # Track tool spans by ID
                tool_spans = {}

                # Call original function and yield messages
                async for message in wrapped(*args, **kwargs):
                    # Parse message for tool uses
                    from claude_agent_sdk import AssistantMessage

                    if isinstance(message, AssistantMessage):
                        # Extract and create tool spans
                        tool_uses = extract_tool_uses(message)
                        for tool_use in tool_uses:
                            tool_span = self._span_manager.start_tool_span(
                                tool_name=tool_use["name"],
                                parent_span=query_span,
                            )
                            # Set tool parameters
                            tool_span.set_attribute(
                                SpanAttributes.TOOL_PARAMETERS,
                                str(tool_use["input"]),
                            )
                            # Track for later closing
                            tool_spans[tool_use["id"]] = tool_span

                    yield message

                # End all tool spans
                for tool_span in tool_spans.values():
                    self._span_manager.end_span(tool_span)

            finally:
                self._span_manager.end_span(query_span)
        finally:
            self._span_manager.end_span(root_span)
