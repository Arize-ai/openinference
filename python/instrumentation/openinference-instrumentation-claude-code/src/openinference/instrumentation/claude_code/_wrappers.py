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
                active_subagent_span = None  # Track active subagent AGENT span
                active_subagent_llm_span = None  # Track active subagent LLM span

                # Call original function and yield messages
                async for message in wrapped(*args, **kwargs):
                    # Parse message for tool uses
                    from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage

                    if isinstance(message, AssistantMessage):
                        # Collect output for span attributes
                        output_messages.append(message)

                        # Capture text content from all assistant messages
                        text_content = extract_text_content(message)
                        tool_uses = extract_tool_uses(message)

                        # Add text content as span event (for visibility in traces)
                        if text_content:
                            response_span.add_event(
                                name="assistant_message",
                                attributes={"content": text_content},
                            )

                        # Extract and create tool spans
                        for tool_use in tool_uses:
                            # Check if this is a Task tool spawning a subagent
                            is_subagent = (
                                tool_use["name"] == "Task"
                                and isinstance(tool_use["input"], dict)
                                and "subagent_type" in tool_use["input"]
                            )

                            if is_subagent:
                                # Create the Task tool span FIRST (before subagent spans)
                                tool_span = self._span_manager.start_tool_span(
                                    tool_name=tool_use["name"],
                                    parent_span=response_span,
                                )
                                tool_span.set_attribute(
                                    SpanAttributes.TOOL_PARAMETERS,
                                    str(tool_use["input"]),
                                )

                                # Now create nested AGENT span for subagent as CHILD of Task tool
                                subagent_name = tool_use["input"]["subagent_type"]
                                subagent_session_id = f"{session_id}-{subagent_name}"

                                active_subagent_span = self._span_manager.start_agent_span(
                                    name=f"Subagent: {subagent_name}",
                                    session_id=subagent_session_id,
                                    parent_span=tool_span,  # Subagent is child of Task tool
                                )

                                # Create LLM span under subagent for its operations
                                active_subagent_llm_span = self._span_manager.start_llm_span(
                                    name=f"Subagent {subagent_name} Operations",
                                    parent_span=active_subagent_span,
                                )

                                # End the Task tool span after subagent completes
                                # (will be ended when subagent ends)
                                tool_spans[tool_use["id"]] = tool_span
                            else:
                                # Regular tool - create under current context
                                # If we're in a subagent, tools go under the subagent's LLM span
                                parent = (
                                    active_subagent_llm_span
                                    if active_subagent_llm_span
                                    else response_span
                                )
                                tool_span = self._span_manager.start_tool_span(
                                    tool_name=tool_use["name"],
                                    parent_span=parent,
                                )
                                # Set tool parameters
                                tool_span.set_attribute(
                                    SpanAttributes.TOOL_PARAMETERS,
                                    str(tool_use["input"]),
                                )
                                # Track for later closing
                                tool_spans[tool_use["id"]] = tool_span

                    elif isinstance(message, UserMessage):
                        # Extract tool results from UserMessage
                        if hasattr(message, "content"):
                            from claude_agent_sdk import ToolResultBlock

                            for block in message.content:
                                if isinstance(block, ToolResultBlock):
                                    # Find the corresponding tool span and set output
                                    tool_span = tool_spans.get(block.tool_use_id)
                                    if tool_span:
                                        # Set tool output
                                        output_content = (
                                            block.content
                                            if isinstance(block.content, str)
                                            else str(block.content)
                                        )
                                        tool_span.set_attribute(
                                            SpanAttributes.OUTPUT_VALUE,
                                            output_content,
                                        )
                                        if block.is_error:
                                            tool_span.set_attribute("tool.is_error", True)

                    elif isinstance(message, ResultMessage):
                        # Extract metadata from ResultMessage and add to root span
                        if hasattr(message, "status"):
                            root_span.set_attribute("result.status", message.status)
                        if hasattr(message, "duration_ms"):
                            root_span.set_attribute("result.duration_ms", message.duration_ms)
                        if hasattr(message, "total_tokens"):
                            root_span.set_attribute("result.total_tokens", message.total_tokens)
                        if hasattr(message, "total_cost_usd"):
                            root_span.set_attribute("result.total_cost_usd", message.total_cost_usd)
                        if hasattr(message, "stop_reason"):
                            root_span.set_attribute("result.stop_reason", message.stop_reason)

                    yield message

                # Set LLM output messages on span - create separate entries for each message
                if output_messages:
                    message_index = 0
                    for msg in output_messages:
                        # Extract text and tool uses from message
                        output_text = extract_text_content(msg)
                        tool_uses = extract_tool_uses(msg)

                        # Build message content including tool calls
                        message_parts = []
                        if output_text:
                            message_parts.append(output_text)

                        # Add tool use information to message content
                        if tool_uses:
                            for tool_use in tool_uses:
                                tool_info = f"[Tool Call: {tool_use['name']}]"
                                message_parts.append(tool_info)

                        if message_parts:  # Only add if there's content
                            content = "\n".join(message_parts)
                            response_span.set_attribute(
                                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.message.role",
                                "assistant",
                            )
                            response_span.set_attribute(
                                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.message.content",
                                content,
                            )
                            # Add tool calls as structured data
                            if tool_uses:
                                for idx, tool_use in enumerate(tool_uses):
                                    response_span.set_attribute(
                                        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.message.tool_calls.{idx}.tool_call.name",
                                        tool_use["name"],
                                    )
                            message_index += 1

                # End all tool spans
                for tool_span in tool_spans.values():
                    self._span_manager.end_span(tool_span)

                # End subagent spans if active
                if active_subagent_llm_span:
                    self._span_manager.end_span(active_subagent_llm_span)
                if active_subagent_span:
                    self._span_manager.end_span(active_subagent_span)

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
        logger.debug("_QueryWrapper called with args=%s, kwargs=%s", args, kwargs)

        # Check suppression
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            logger.debug("Tracing suppressed, passing through")
            async for message in wrapped(*args, **kwargs):
                yield message
            return

        # Extract prompt from kwargs
        prompt = kwargs.get("prompt", "")
        session_id = f"query-{id(prompt)}"  # Generate session ID
        logger.debug("Creating query session with session_id=%s", session_id)

        # Start root AGENT span
        root_span = self._span_manager.start_agent_span(
            name="Claude Code Query Session",
            session_id=session_id,
        )
        logger.debug("Started root span: %s", root_span)

        try:
            # Start LLM span for query
            query_span = self._span_manager.start_llm_span(
                name="Claude Code Query",
                parent_span=root_span,
            )
            logger.debug("Started query LLM span: %s", query_span)

            # Set input message from prompt
            if prompt:
                query_span.set_attribute(
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role",
                    "user",
                )
                query_span.set_attribute(
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.content",
                    prompt,
                )

            try:
                # Track tool spans by ID and collect output
                tool_spans = {}
                output_messages = []
                active_subagent_span = None  # Track active subagent AGENT span
                active_subagent_llm_span = None  # Track active subagent LLM span
                message_count = 0

                # Call original function and yield messages
                async for message in wrapped(*args, **kwargs):
                    message_count += 1
                    logger.debug("Received message #%d: %s", message_count, type(message).__name__)

                    # Parse message for tool uses
                    from claude_agent_sdk import AssistantMessage, ResultMessage, UserMessage

                    if isinstance(message, AssistantMessage):
                        # Collect output for span attributes
                        output_messages.append(message)

                        # Capture text content from all assistant messages
                        text_content = extract_text_content(message)
                        tool_uses = extract_tool_uses(message)

                        # Add text content as span event (for visibility in traces)
                        if text_content:
                            query_span.add_event(
                                name="assistant_message",
                                attributes={"content": text_content},
                            )
                            logger.debug("Added assistant message text as span event")

                        # Extract and create tool spans
                        logger.debug("Extracted %d tool uses from message", len(tool_uses))
                        for tool_use in tool_uses:
                            # Check if this is a Task tool spawning a subagent
                            is_subagent = (
                                tool_use["name"] == "Task"
                                and isinstance(tool_use["input"], dict)
                                and "subagent_type" in tool_use["input"]
                            )

                            if is_subagent:
                                # Create the Task tool span FIRST (before subagent spans)
                                tool_span = self._span_manager.start_tool_span(
                                    tool_name=tool_use["name"],
                                    parent_span=query_span,
                                )
                                logger.debug("Started tool span for %s: %s", tool_use["name"], tool_span)
                                tool_span.set_attribute(
                                    SpanAttributes.TOOL_PARAMETERS,
                                    str(tool_use["input"]),
                                )

                                # Now create nested AGENT span for subagent as CHILD of Task tool
                                subagent_name = tool_use["input"]["subagent_type"]
                                subagent_session_id = f"{session_id}-{subagent_name}"

                                active_subagent_span = self._span_manager.start_agent_span(
                                    name=f"Subagent: {subagent_name}",
                                    session_id=subagent_session_id,
                                    parent_span=tool_span,  # Subagent is child of Task tool
                                )

                                # Create LLM span under subagent for its operations
                                active_subagent_llm_span = self._span_manager.start_llm_span(
                                    name=f"Subagent {subagent_name} Operations",
                                    parent_span=active_subagent_span,
                                )

                                # End the Task tool span after subagent completes
                                # (will be ended when subagent ends)
                                tool_spans[tool_use["id"]] = tool_span
                            else:
                                # Regular tool - create under current context
                                # If we're in a subagent, tools go under the subagent's LLM span
                                parent = (
                                    active_subagent_llm_span
                                    if active_subagent_llm_span
                                    else query_span
                                )
                                tool_span = self._span_manager.start_tool_span(
                                    tool_name=tool_use["name"],
                                    parent_span=parent,
                                )
                                logger.debug("Started tool span for %s: %s", tool_use["name"], tool_span)
                                # Set tool parameters
                                tool_span.set_attribute(
                                    SpanAttributes.TOOL_PARAMETERS,
                                    str(tool_use["input"]),
                                )
                                # Track for later closing
                                tool_spans[tool_use["id"]] = tool_span

                    elif isinstance(message, UserMessage):
                        # Extract tool results from UserMessage
                        if hasattr(message, "content"):
                            from claude_agent_sdk import ToolResultBlock

                            for block in message.content:
                                if isinstance(block, ToolResultBlock):
                                    # Find the corresponding tool span and set output
                                    tool_span = tool_spans.get(block.tool_use_id)
                                    if tool_span:
                                        # Set tool output
                                        output_content = (
                                            block.content
                                            if isinstance(block.content, str)
                                            else str(block.content)
                                        )
                                        tool_span.set_attribute(
                                            SpanAttributes.OUTPUT_VALUE,
                                            output_content,
                                        )
                                        if block.is_error:
                                            tool_span.set_attribute("tool.is_error", True)
                                        logger.debug(
                                            "Set tool output for tool_use_id: %s", block.tool_use_id
                                        )

                    elif isinstance(message, ResultMessage):
                        # Extract metadata from ResultMessage and add to root span
                        logger.debug("Received ResultMessage with metadata")
                        if hasattr(message, "status"):
                            root_span.set_attribute("result.status", message.status)
                        if hasattr(message, "duration_ms"):
                            root_span.set_attribute("result.duration_ms", message.duration_ms)
                        if hasattr(message, "total_tokens"):
                            root_span.set_attribute("result.total_tokens", message.total_tokens)
                        if hasattr(message, "total_cost_usd"):
                            root_span.set_attribute("result.total_cost_usd", message.total_cost_usd)
                        # Add stop_reason if available
                        if hasattr(message, "stop_reason"):
                            root_span.set_attribute("result.stop_reason", message.stop_reason)

                    yield message

                logger.debug("Finished processing %d messages, ending %d tool spans", message_count, len(tool_spans))

                # Set LLM output messages on span - create separate entries for each message
                if output_messages:
                    message_index = 0
                    for msg in output_messages:
                        # Extract text and tool uses from message
                        output_text = extract_text_content(msg)
                        tool_uses = extract_tool_uses(msg)

                        # Build message content including tool calls
                        message_parts = []
                        if output_text:
                            message_parts.append(output_text)

                        # Add tool use information to message content
                        if tool_uses:
                            for tool_use in tool_uses:
                                tool_info = f"[Tool Call: {tool_use['name']}]"
                                message_parts.append(tool_info)

                        if message_parts:  # Only add if there's content
                            content = "\n".join(message_parts)
                            query_span.set_attribute(
                                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.message.role",
                                "assistant",
                            )
                            query_span.set_attribute(
                                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.message.content",
                                content,
                            )
                            # Add tool calls as structured data
                            if tool_uses:
                                for idx, tool_use in enumerate(tool_uses):
                                    query_span.set_attribute(
                                        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.message.tool_calls.{idx}.tool_call.name",
                                        tool_use["name"],
                                    )
                            message_index += 1

                # End all tool spans
                for tool_span in tool_spans.values():
                    self._span_manager.end_span(tool_span)

                # End subagent spans if active
                if active_subagent_llm_span:
                    self._span_manager.end_span(active_subagent_llm_span)
                if active_subagent_span:
                    self._span_manager.end_span(active_subagent_span)

            finally:
                logger.debug("Ending query LLM span")
                self._span_manager.end_span(query_span)
        finally:
            logger.debug("Ending root AGENT span")
            self._span_manager.end_span(root_span)
