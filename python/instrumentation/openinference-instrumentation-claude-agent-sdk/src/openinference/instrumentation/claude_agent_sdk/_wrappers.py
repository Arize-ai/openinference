"""Wrappers for Claude Agent SDK query() to produce OpenInference spans."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import TYPE_CHECKING, Any, Callable, Mapping, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api

from openinference.instrumentation import get_attributes_from_context, safe_json_dumps
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL

if TYPE_CHECKING:
    from claude_agent_sdk.types import Message

AGENT = OpenInferenceSpanKindValues.AGENT.value
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
JSON = OpenInferenceMimeTypeValues.JSON.value
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_TOOLS = SpanAttributes.LLM_TOOLS
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA


def _query_input_value(prompt: Any, options: Any) -> str:
    """Build a JSON-serializable input description for the span."""
    payload: dict[str, Any] = {}
    if isinstance(prompt, str):
        payload["prompt"] = prompt
    elif prompt is None:
        payload["prompt"] = None
    else:
        payload["prompt"] = "<AsyncIterable>"
    if options is not None:
        try:
            payload["options"] = options.model_dump() if hasattr(options, "model_dump") else {}
        except Exception:
            payload["options"] = {}
    return safe_json_dumps(payload)


def _iter_tool_attributes(options: Any) -> Iterator[tuple[str, Any]]:
    """Yield OpenInference llm.tools attributes from ClaudeAgentOptions.

    Includes allowed_tools and tools (list of names). MCP server tools
    (e.g. mcp__server__tool_name from allowed_tools) are captured as-is.
    """
    if options is None:
        return
    seen: set[str] = set()
    tool_names: list[str] = []
    for attr in ("allowed_tools", "tools"):
        if not hasattr(options, attr):
            continue
        val = getattr(options, attr, None)
        if not isinstance(val, list):
            continue
        for t in val:
            name = t if isinstance(t, str) else (t.get("name") if isinstance(t, dict) else None)
            if isinstance(name, str) and name not in seen:
                seen.add(name)
                tool_names.append(name)
    for idx, name in enumerate(tool_names):
        schema = {
            "type": "function",
            "function": {"name": name, "description": f"Tool: {name}"},
        }
        yield f"{LLM_TOOLS}.{idx}.{TOOL_JSON_SCHEMA}", safe_json_dumps(schema)


def _message_content_str(m: Any) -> str:
    """Extract a string representation of message content for display."""
    if hasattr(m, "result") and getattr(m, "result") is not None:
        return str(m.result)
    if hasattr(m, "content"):
        content = getattr(m, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(getattr(block, "text", ""))
                elif hasattr(block, "thinking"):
                    parts.append(f"[thinking] {getattr(block, 'thinking', '')}")
                elif hasattr(block, "name") and hasattr(block, "input"):
                    parts.append(
                        f"[tool: {getattr(block, 'name', '')}] "
                        f"{safe_json_dumps(getattr(block, 'input', {}))}"
                    )
                elif hasattr(block, "tool_use_id") and hasattr(block, "content"):
                    parts.append(f"[tool_result] {getattr(block, 'content', '')}")
            return "\n".join(parts) if parts else ""
    if hasattr(m, "subtype") and hasattr(m, "data"):
        return f"subtype={getattr(m, 'subtype')} data={safe_json_dumps(getattr(m, 'data', {}))}"
    if hasattr(m, "event"):
        return safe_json_dumps(getattr(m, "event", {}))
    return ""


def _message_role(m: Any) -> str:
    """Map SDK message to OpenInference message role."""
    name = type(m).__name__
    if name == "UserMessage":
        return "user"
    if name in ("AssistantMessage", "ResultMessage"):
        return "assistant"
    if name == "SystemMessage":
        return "system"
    if name == "StreamEvent":
        return "assistant"
    return "unknown"


def _iter_output_message_attributes(messages: list[Any]) -> Iterator[tuple[str, Any]]:
    """Yield OpenInference flattened llm.output_messages attributes per message."""
    for idx, m in enumerate(messages):
        role = _message_role(m)
        content = _message_content_str(m)
        prefix = f"{LLM_OUTPUT_MESSAGES}.{idx}"
        yield f"{prefix}.{MESSAGE_ROLE}", role
        if content:
            yield f"{prefix}.{MESSAGE_CONTENT}", content
        # Tool use blocks on AssistantMessage
        if hasattr(m, "content") and isinstance(getattr(m, "content"), list):
            tool_idx = 0
            for block in getattr(m, "content", []):
                if hasattr(block, "name") and hasattr(block, "input"):
                    yield (
                        f"{prefix}.{MESSAGE_TOOL_CALLS}.{tool_idx}.{TOOL_CALL_FUNCTION_NAME}",
                        getattr(block, "name", ""),
                    )
                    yield (
                        f"{prefix}.{MESSAGE_TOOL_CALLS}.{tool_idx}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        safe_json_dumps(getattr(block, "input", {})),
                    )
                    tool_idx += 1


def _safe_int(value: Any) -> int | None:
    """Coerce value to int for token counts; return None if not possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_usage_attributes(messages: list[Any]) -> Iterator[tuple[str, int]]:
    """Yield OpenInference token-count attributes from the last ResultMessage.usage.

    ResultMessage.usage is a dict with keys like input_tokens, output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens (Anthropic-style).
    Uses the last message that has a dict-like usage attribute.
    """
    usage: dict[str, Any] | None = None
    for m in reversed(messages):
        u = getattr(m, "usage", None)
        if isinstance(u, dict):
            usage = u
            break
    if not usage:
        return
    input_tokens = _safe_int(usage.get("input_tokens"))
    output_tokens = _safe_int(usage.get("output_tokens"))
    cache_read = _safe_int(usage.get("cache_read_input_tokens"))
    cache_write = _safe_int(usage.get("cache_creation_input_tokens"))
    if input_tokens is not None:
        yield LLM_TOKEN_COUNT_PROMPT, input_tokens
    if output_tokens is not None:
        yield LLM_TOKEN_COUNT_COMPLETION, output_tokens
    if input_tokens is not None and output_tokens is not None:
        yield LLM_TOKEN_COUNT_TOTAL, input_tokens + output_tokens
    if cache_read is not None and cache_read != 0:
        yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ, cache_read
    if cache_write is not None and cache_write != 0:
        yield LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE, cache_write


def _output_value_from_messages(messages: list[Any]) -> str:
    """Return the actual LLM answer: ResultMessage.result, or fallback to summary."""
    for m in reversed(messages):
        result = getattr(m, "result", None)
        if result is not None and isinstance(result, str):
            return str(result)
    return _messages_summary(messages)


def _messages_summary(messages: list[Any]) -> str:
    """Summarize collected messages when no ResultMessage.result is available."""
    if not messages:
        return "[]"
    try:
        summary = []
        for m in messages:
            item: dict[str, Any] = {}
            if hasattr(m, "type"):
                item["type"] = getattr(m, "type", None)
            else:
                item["type"] = type(m).__name__
            if hasattr(m, "subtype"):
                item["subtype"] = getattr(m, "subtype", None)
            if hasattr(m, "result") and getattr(m, "result") is not None:
                item["has_result"] = True
            summary.append(item)
        return safe_json_dumps(summary)
    except Exception:
        return safe_json_dumps({"message_count": len(messages)})


class _QueryWrapper:
    """Wraps the async generator returned by query() to trace the full agent run."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncIterator["Message"]:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for message in wrapped(*args, **kwargs):
                yield message
            return

        prompt = kwargs.get("prompt")
        options = kwargs.get("options")

        span = self._tracer.start_span(
            "ClaudeAgentSDK.query",
            attributes=dict(
                [
                    (OPENINFERENCE_SPAN_KIND, AGENT),
                    (INPUT_VALUE, _query_input_value(prompt, options)),
                    (INPUT_MIME_TYPE, JSON),
                ]
                + list(_iter_tool_attributes(options))
                + list(get_attributes_from_context())
            ),
        )

        ctx = trace_api.set_span_in_context(span)
        token = context_api.attach(ctx)
        messages: list[Any] = []

        try:
            async for message in wrapped(*args, **kwargs):
                messages.append(message)
                yield message
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            # Descriptive status for SDK errors (ClaudeSDKError, ProcessError, etc.)
            err_msg = f"{type(exc).__name__}: {exc}"
            span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, err_msg))
            raise
        finally:
            try:
                span.set_attribute(OUTPUT_VALUE, _output_value_from_messages(messages))
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                for key, value in _iter_output_message_attributes(messages):
                    span.set_attribute(key, value)
                for key, value in _iter_usage_attributes(messages):
                    span.set_attribute(key, value)
                if span.is_recording() and span.status.status_code != trace_api.StatusCode.ERROR:  # type: ignore[attr-defined]
                    span.set_status(trace_api.StatusCode.OK)
            finally:
                span.end()
                context_api.detach(token)


# Attributes on client instance to link query() input to receive_response() span
_OINFERENCE_LAST_PROMPT = "_oinference_last_prompt"
_OINFERENCE_LAST_OPTIONS = "_oinference_last_options"


class _ClientQueryWrapper:
    """Records prompt/options on the client for the next receive_response() span."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        prompt = kwargs.get("prompt") if kwargs else (args[0] if args else None)
        setattr(instance, _OINFERENCE_LAST_PROMPT, prompt)
        setattr(
            instance,
            _OINFERENCE_LAST_OPTIONS,
            getattr(instance, "options", None),
        )
        return await wrapped(*args, **kwargs)


class _ClientConnectWrapper:
    """Records initial prompt/options from connect() for the first receive_response() span."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        prompt = kwargs.get("prompt") if kwargs else (args[0] if args else None)
        if prompt is not None:
            setattr(instance, _OINFERENCE_LAST_PROMPT, prompt)
            setattr(
                instance,
                _OINFERENCE_LAST_OPTIONS,
                getattr(instance, "options", None),
            )
        return await wrapped(*args, **kwargs)


class _ClientReceiveResponseWrapper:
    """Wraps receive_response() to create an AGENT span per response turn."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    async def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: Tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> AsyncIterator["Message"]:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            async for message in wrapped(*args, **kwargs):
                yield message
            return

        prompt = getattr(instance, _OINFERENCE_LAST_PROMPT, None)
        options = getattr(instance, _OINFERENCE_LAST_OPTIONS, None)

        span = self._tracer.start_span(
            "ClaudeAgentSDK.ClaudeSDKClient.receive_response",
            attributes=dict(
                [
                    (OPENINFERENCE_SPAN_KIND, AGENT),
                    (INPUT_VALUE, _query_input_value(prompt, options)),
                    (INPUT_MIME_TYPE, JSON),
                ]
                + list(_iter_tool_attributes(options))
                + list(get_attributes_from_context())
            ),
        )
        ctx = trace_api.set_span_in_context(span)
        token = context_api.attach(ctx)
        messages: list[Any] = []

        try:
            async for message in wrapped(*args, **kwargs):
                messages.append(message)
                yield message
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            err_msg = f"{type(exc).__name__}: {exc}"
            span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, err_msg))
            raise
        finally:
            try:
                span.set_attribute(OUTPUT_VALUE, _output_value_from_messages(messages))
                span.set_attribute(OUTPUT_MIME_TYPE, JSON)
                for key, value in _iter_output_message_attributes(messages):
                    span.set_attribute(key, value)
                for key, value in _iter_usage_attributes(messages):
                    span.set_attribute(key, value)
                if span.is_recording() and span.status.status_code != trace_api.StatusCode.ERROR:  # type: ignore[attr-defined]
                    span.set_status(trace_api.StatusCode.OK)
            finally:
                span.end()
                context_api.detach(token)
