"""Wrappers for Claude Agent SDK query() to produce OpenInference spans."""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator
from collections.abc import Mapping as MappingABC
from typing import TYPE_CHECKING, Any, Callable, Mapping, Tuple

import opentelemetry.context as context_api
from opentelemetry import trace as trace_api

from openinference.instrumentation import (
    get_attributes_from_context,
    get_input_attributes,
    get_output_attributes,
    get_tool_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from claude_agent_sdk.types import Message

AGENT = OpenInferenceSpanKindValues.AGENT.value
TOOL = OpenInferenceSpanKindValues.TOOL.value
JSON = OpenInferenceMimeTypeValues.JSON
TEXT = OpenInferenceMimeTypeValues.TEXT

OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
SESSION_ID = SpanAttributes.SESSION_ID
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_COST_TOTAL = SpanAttributes.LLM_COST_TOTAL
AGENT_NAME = SpanAttributes.AGENT_NAME
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_SYSTEM_ANTHROPIC = OpenInferenceLLMSystemValues.ANTHROPIC.value
TOOL_ID = SpanAttributes.TOOL_ID
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, MappingABC):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_record(value: Any) -> dict[str, Any]:
    if isinstance(value, MappingABC):
        return dict(value)
    return {}


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_usage(usage: Any) -> Mapping[str, Any]:
    if isinstance(usage, MappingABC):
        return usage
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        "cache_write_input_tokens": getattr(usage, "cache_write_input_tokens", None),
    }


def _format_prompt_attributes(prompt: Any) -> dict[str, Any]:
    if prompt is None:
        return {}
    if isinstance(prompt, str):
        return get_input_attributes(prompt, mime_type=TEXT)
    return get_input_attributes(safe_json_dumps(prompt), mime_type=JSON)


def _is_system_init_message(msg: Any) -> bool:
    msg_type = _get_field(msg, "type")
    subtype = _get_field(msg, "subtype")
    if msg_type == "system" and subtype == "init":
        return True
    if subtype == "init" and (_get_field(msg, "session_id") or _get_field(msg, "model")):
        return True
    return False


def _extract_model_name_from_usage(model_usage: Any) -> str | None:
    if isinstance(model_usage, MappingABC) and model_usage:
        return str(next(iter(model_usage.keys())))
    if isinstance(model_usage, (list, tuple)):
        for entry in model_usage:
            name = (
                _get_field(entry, "model")
                or _get_field(entry, "name")
                or _get_field(entry, "model_name")
            )
            if name:
                return str(name)
    if model_usage is not None:
        name = (
            _get_field(model_usage, "model")
            or _get_field(model_usage, "name")
            or _get_field(model_usage, "model_name")
        )
        if name:
            return str(name)
    return None


def _extract_model_name(msg: Any) -> str | None:
    raw_name = _get_field(msg, "model") or _get_field(msg, "model_name")
    if raw_name:
        return str(raw_name)
    model_usage = _get_field(msg, "modelUsage") or _get_field(msg, "model_usage")
    usage_name = _extract_model_name_from_usage(model_usage)
    if usage_name:
        return usage_name
    usage = _get_field(msg, "usage")
    model_usage = _get_field(usage, "modelUsage") or _get_field(usage, "model_usage")
    usage_name = _extract_model_name_from_usage(model_usage)
    if usage_name:
        return usage_name
    raw_name = _get_field(usage, "model") or _get_field(usage, "model_name")
    if raw_name:
        return str(raw_name)
    data = _get_field(msg, "data", {})
    raw_name = _get_field(data, "model") or _get_field(data, "model_name")
    if raw_name:
        return str(raw_name)
    return None


def _maybe_set_model(span: trace_api.Span, msg: Any) -> None:
    if not span.is_recording():
        return
    model = _extract_model_name(msg)
    if not model:
        inner = _get_field(msg, "message")
        model = _extract_model_name(inner)
    if model:
        span.set_attribute(LLM_MODEL_NAME, model)


def _extract_init_attributes(msg: Any) -> dict[str, Any]:
    session_id = _get_field(msg, "session_id")
    if session_id is None:
        session_id = _get_field(_get_field(msg, "data", {}), "session_id")
    model = _extract_model_name(msg)
    attributes: dict[str, Any] = {}
    if session_id:
        attributes[SESSION_ID] = session_id
    if model:
        attributes[LLM_MODEL_NAME] = model
    return attributes


def _is_result_success_message(msg: Any) -> bool:
    msg_type = _get_field(msg, "type")
    subtype = _get_field(msg, "subtype")
    if msg_type == "result" and subtype == "success" and not _get_field(msg, "is_error"):
        return True
    return (
        subtype == "success"
        and _get_field(msg, "usage") is not None
        and not _get_field(msg, "is_error")
    )


def _is_result_error_message(msg: Any) -> bool:
    # For SDK typed ResultMessage objects: use the authoritative is_error field.
    if _get_field(msg, "is_error") is True:
        return True
    msg_type = _get_field(msg, "type")
    subtype = _get_field(msg, "subtype")
    if msg_type == "result" and isinstance(subtype, str) and subtype.startswith("error"):
        return True
    return (
        isinstance(subtype, str)
        and subtype.startswith("error")
        and _get_field(msg, "usage") is not None
    )


def _extract_usage_and_cost_attributes(msg: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if model := _extract_model_name(msg):
        attributes[LLM_MODEL_NAME] = model
    usage = _coerce_usage(_get_field(msg, "usage"))
    input_tokens = _safe_int(usage.get("input_tokens"))
    output_tokens = _safe_int(usage.get("output_tokens"))
    cache_read_tokens = _safe_int(usage.get("cache_read_input_tokens"))
    cache_write_tokens = _safe_int(
        usage.get("cache_write_input_tokens")
        if usage.get("cache_write_input_tokens") is not None
        else usage.get("cache_creation_input_tokens")
    )
    if input_tokens is not None:
        attributes[LLM_TOKEN_COUNT_PROMPT] = input_tokens
    if output_tokens is not None:
        attributes[LLM_TOKEN_COUNT_COMPLETION] = output_tokens
    if input_tokens is not None and output_tokens is not None:
        attributes[LLM_TOKEN_COUNT_TOTAL] = input_tokens + output_tokens
    if cache_read_tokens is not None:
        attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] = cache_read_tokens
    if cache_write_tokens is not None:
        attributes[LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] = cache_write_tokens
    if (cost := _safe_float(_get_field(msg, "total_cost_usd"))) is not None:
        attributes[LLM_COST_TOTAL] = cost
    if session_id := _get_field(msg, "session_id"):
        attributes[SESSION_ID] = session_id
    return attributes


def _extract_result_success_attributes(msg: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    if (result := _get_field(msg, "result")) is not None:
        attributes.update(get_output_attributes(result))
    attributes.update(_extract_usage_and_cost_attributes(msg))
    return attributes


def _extract_result_error_attributes(msg: Any) -> dict[str, Any]:
    attributes: dict[str, Any] = {}
    errors = _get_field(msg, "errors")
    if errors:
        attributes.update(get_output_attributes(safe_json_dumps(errors), mime_type=JSON))
    attributes.update(_extract_usage_and_cost_attributes(msg))
    return attributes


def _process_message(msg: Any, span: trace_api.Span) -> bool:
    _maybe_set_model(span, msg)
    if _is_system_init_message(msg):
        span.set_attributes(_extract_init_attributes(msg))
        return False
    if _is_result_success_message(msg):
        span.set_attributes(_extract_result_success_attributes(msg))
        return False
    if _is_result_error_message(msg):
        span.set_attributes(_extract_result_error_attributes(msg))
        subtype = _get_field(msg, "subtype")
        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, f"Result error: {subtype}"))
        return True
    return False


def _make_hook_matcher(callback: Callable[[Any], Any]) -> Any:
    for module_path, name in (
        ("claude_agent_sdk", "HookMatcher"),
        ("claude_agent_sdk.types", "HookMatcher"),
        ("claude_agent_sdk.types", "HookCallbackMatcher"),
    ):
        try:
            module = __import__(module_path, fromlist=[name])
            matcher_type = getattr(module, name)
        except Exception:
            matcher_type = None
        if matcher_type is None:
            continue
        try:
            # HookMatcher is a @dataclass (not a Mapping). Return it directly so
            # the SDK's _convert_hooks_to_internal_format can see hasattr(m, "hooks").
            return matcher_type(hooks=[callback])
        except Exception:
            continue
    return {"hooks": [callback]}


def _hook_event_name(payload: Any) -> str | None:
    value = _get_field(payload, "hook_event_name")
    if value is None:
        return None
    return str(value)


def _create_tool_hook_matchers(
    tool_tracker: "_ToolSpanTrackerBase",
) -> dict[str, list[Any]]:
    async def pre_tool_use(
        input_data: Any,
        tool_use_id: Any | None = None,
        context: Any | None = None,
    ) -> dict[str, Any]:
        del context
        try:
            if (name := _hook_event_name(input_data)) and name != "PreToolUse":
                return {}
            tool_name = _get_field(input_data, "tool_name")
            tool_input = _get_field(input_data, "tool_input")
            resolved_tool_use_id = tool_use_id or _get_field(input_data, "tool_use_id")
            parent_tool_use_id = _get_field(input_data, "parent_tool_use_id")
            tool_tracker.start_tool_span(
                tool_name,
                tool_input,
                resolved_tool_use_id,
                parent_tool_use_id,
            )
        except Exception:
            pass
        return {}

    async def post_tool_use(
        input_data: Any,
        tool_use_id: Any | None = None,
        context: Any | None = None,
    ) -> dict[str, Any]:
        del context
        try:
            if (name := _hook_event_name(input_data)) and name != "PostToolUse":
                return {}
            resolved_tool_use_id = tool_use_id or _get_field(input_data, "tool_use_id")
            tool_response = _get_field(input_data, "tool_response")
            tool_tracker.end_tool_span(resolved_tool_use_id, tool_response)
        except Exception:
            pass
        return {}

    async def post_tool_use_failure(
        input_data: Any,
        tool_use_id: Any | None = None,
        context: Any | None = None,
    ) -> dict[str, Any]:
        del context
        try:
            if (name := _hook_event_name(input_data)) and name != "PostToolUseFailure":
                return {}
            resolved_tool_use_id = tool_use_id or _get_field(input_data, "tool_use_id")
            error = _get_field(input_data, "error")
            tool_tracker.end_tool_span_with_error(resolved_tool_use_id, error)
        except Exception:
            pass
        return {}

    return {
        "PreToolUse": [_make_hook_matcher(pre_tool_use)],
        "PostToolUse": [_make_hook_matcher(post_tool_use)],
        "PostToolUseFailure": [_make_hook_matcher(post_tool_use_failure)],
    }


def _get_hooks(options: Any) -> Mapping[str, Any] | None:
    if options is None:
        return None
    if isinstance(options, MappingABC):
        return options.get("hooks")
    hooks = getattr(options, "hooks", None)
    if isinstance(hooks, MappingABC):
        return hooks
    if hooks is None:
        return None
    hook_events = (
        "SessionStart",
        "PreToolUse",
        "PostToolUse",
        "PostToolUseFailure",
        "Stop",
        "SendError",
    )
    extracted: dict[str, Any] = {}
    for event in hook_events:
        if hasattr(hooks, event):
            extracted[event] = getattr(hooks, event)
    if extracted:
        return extracted
    if hasattr(hooks, "__dict__"):
        data = {k: v for k, v in getattr(hooks, "__dict__", {}).items() if not k.startswith("_")}
        if isinstance(data, MappingABC):
            return data
    for attr in ("model_dump", "dict"):
        dump = getattr(hooks, attr, None)
        if callable(dump):
            try:
                dumped = dump()
                if isinstance(dumped, MappingABC):
                    return dumped
            except Exception:
                pass
    return None


def _set_hooks(options: Any, hooks: Mapping[str, Any]) -> Any:
    if isinstance(options, MappingABC):
        return {**options, "hooks": hooks}

    try:
        new_options = copy.copy(options)
        setattr(new_options, "hooks", hooks)
        return new_options
    except Exception:
        pass
    try:
        setattr(options, "hooks", hooks)
        return options
    except Exception:
        pass
    if hasattr(options, "model_copy"):
        try:
            return options.model_copy(update={"hooks": hooks})
        except Exception:
            pass
    if hasattr(options, "copy"):
        try:
            return options.copy(update={"hooks": hooks})
        except Exception:
            pass
    return options


def _ensure_options(options: Any) -> Any | None:
    if options is not None:
        return options
    try:
        from claude_agent_sdk.types import ClaudeAgentOptions

        return ClaudeAgentOptions()
    except Exception:
        return None


def _merge_hooks(options: Any, tool_tracker: "_ToolSpanTrackerBase") -> Any | None:
    opts = _ensure_options(options)
    if opts is None:
        return None

    existing_hooks = _get_hooks(opts)
    if not isinstance(existing_hooks, MappingABC):
        existing_hooks = {}
    merged_hooks: dict[str, Any] = dict(existing_hooks)
    our_hooks = _create_tool_hook_matchers(tool_tracker)
    for event, matchers in our_hooks.items():
        current = merged_hooks.get(event, [])
        if not isinstance(current, list):
            current = [current]
        normalized: list[Any] = []
        for matcher in current:
            if isinstance(matcher, MappingABC):
                normalized.append(dict(matcher))
                continue
            # Keep HookMatcher (and any other non-Mapping) as-is.
            # The SDK's _convert_hooks_to_internal_format uses hasattr(m, "hooks")
            # which works on HookMatcher dataclasses but not on plain dicts.
            normalized.append(matcher)
        merged_hooks[event] = [*normalized, *matchers]
    return _set_hooks(opts, merged_hooks)


def _extract_prompt_and_options(
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[Any, Any]:
    prompt = kwargs.get("prompt") if kwargs else None
    options = kwargs.get("options") if kwargs else None
    if prompt is None and args:
        prompt = args[0]
    if options is None and len(args) > 1:
        options = args[1]
    return prompt, options


def _apply_options(
    args: Tuple[Any, ...],
    kwargs: Mapping[str, Any],
    options: Any,
) -> tuple[Tuple[Any, ...], dict[str, Any]]:
    new_kwargs = dict(kwargs)
    if "options" in new_kwargs or len(args) < 2:
        new_kwargs["options"] = options
        return args, new_kwargs
    new_args = list(args)
    if len(new_args) > 1:
        new_args[1] = options
    return tuple(new_args), new_kwargs


class _ToolSpanTrackerBase:
    def start_tool_span(
        self,
        tool_name: Any,
        tool_input: Any,
        tool_use_id: Any,
        parent_tool_use_id: Any = None,
    ) -> None:
        raise NotImplementedError

    def end_tool_span(self, tool_use_id: Any, tool_response: Any) -> None:
        raise NotImplementedError

    def end_tool_span_with_error(self, tool_use_id: Any, error: Any) -> None:
        raise NotImplementedError

    def end_all_in_flight(self) -> None:
        raise NotImplementedError


class _ToolSpanTracker(_ToolSpanTrackerBase):
    def __init__(
        self,
        tracer: trace_api.Tracer,
        parent_span: trace_api.Span | None,
        parent_span_resolver: Callable[[Any], trace_api.Span | None] | None = None,
    ) -> None:
        self._tracer = tracer
        self._parent_span = parent_span
        self._parent_span_resolver = parent_span_resolver
        self._in_flight: dict[str, trace_api.Span] = {}
        self._tool_names: dict[str, str] = {}

    def start_tool_span(
        self,
        tool_name: Any,
        tool_input: Any,
        tool_use_id: Any,
        parent_tool_use_id: Any = None,
    ) -> None:
        if not tool_use_id or not tool_name:
            return
        tool_use_key = str(tool_use_id)
        if tool_use_key in self._in_flight:
            return
        tool_name_str = str(tool_name)
        tool_params = _as_record(tool_input)
        attributes = {
            OPENINFERENCE_SPAN_KIND: TOOL,
            TOOL_ID: tool_use_key,
            **get_tool_attributes(name=tool_name_str, parameters=tool_params),
            **get_input_attributes(safe_json_dumps(tool_input), mime_type=JSON),
        }
        parent_span = self._parent_span
        if (
            parent_tool_use_id is not None
            and str(parent_tool_use_id) != ""
            and self._parent_span_resolver is not None
        ):
            resolved = self._parent_span_resolver(parent_tool_use_id)
            if resolved is not None:
                parent_span = resolved
        ctx = trace_api.set_span_in_context(parent_span) if parent_span is not None else None
        span = self._tracer.start_span(
            tool_name_str,
            context=ctx,
            attributes=attributes,
        )
        self._in_flight[tool_use_key] = span
        self._tool_names[tool_use_key] = tool_name_str

    def end_tool_span(self, tool_use_id: Any, tool_response: Any) -> None:
        if tool_use_id is None:
            return
        tool_use_key = str(tool_use_id)
        span = self._in_flight.pop(tool_use_key, None)
        self._tool_names.pop(tool_use_key, None)
        if span is None:
            return
        if tool_response is not None:
            span.set_attributes(
                get_output_attributes(safe_json_dumps(tool_response), mime_type=JSON)
            )
        span.set_status(trace_api.Status(trace_api.StatusCode.OK))
        span.end()

    def end_tool_span_with_error(self, tool_use_id: Any, error: Any) -> None:
        if tool_use_id is None:
            return
        tool_use_key = str(tool_use_id)
        span = self._in_flight.pop(tool_use_key, None)
        self._tool_names.pop(tool_use_key, None)
        if span is None:
            return
        error_msg = str(error) if error is not None else "Tool execution error"
        span.record_exception(Exception(error_msg))
        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, error_msg))
        span.end()

    def end_all_in_flight(self) -> None:
        for span in self._in_flight.values():
            try:
                span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, "Abandoned"))
                span.end()
            except Exception:
                pass
        self._in_flight.clear()
        self._tool_names.clear()

    def get_in_flight_span(self, tool_use_id: Any) -> trace_api.Span | None:
        if tool_use_id is None:
            return None
        return self._in_flight.get(str(tool_use_id))

    def get_tool_name(self, tool_use_id: Any) -> str | None:
        if tool_use_id is None:
            return None
        return self._tool_names.get(str(tool_use_id))


class _DelegatingToolSpanTracker(_ToolSpanTrackerBase):
    def __init__(self) -> None:
        self._delegate: _ToolSpanTracker | None = None

    def set_delegate(self, tracker: _ToolSpanTracker) -> None:
        self._delegate = tracker

    def clear_delegate(self) -> None:
        self._delegate = None

    def start_tool_span(
        self,
        tool_name: Any,
        tool_input: Any,
        tool_use_id: Any,
        parent_tool_use_id: Any = None,
    ) -> None:
        if self._delegate is not None:
            self._delegate.start_tool_span(
                tool_name,
                tool_input,
                tool_use_id,
                parent_tool_use_id,
            )

    def end_tool_span(self, tool_use_id: Any, tool_response: Any) -> None:
        if self._delegate is not None:
            self._delegate.end_tool_span(tool_use_id, tool_response)

    def end_tool_span_with_error(self, tool_use_id: Any, error: Any) -> None:
        if self._delegate is not None:
            self._delegate.end_tool_span_with_error(tool_use_id, error)

    def end_all_in_flight(self) -> None:
        if self._delegate is not None:
            self._delegate.end_all_in_flight()


# Attributes on client instance to link query() input to receive_response() span
_OINFERENCE_LAST_PROMPT = "_oinference_last_prompt"
_OINFERENCE_HOOKS_INJECTED = "_oinference_hooks_injected"
_OINFERENCE_DELEGATING_TRACKER = "_oinference_delegating_tracker"


def _get_or_create_delegating_tracker(instance: Any) -> _DelegatingToolSpanTracker:
    delegating_tracker = getattr(instance, _OINFERENCE_DELEGATING_TRACKER, None)
    if delegating_tracker is None:
        delegating_tracker = _DelegatingToolSpanTracker()
        setattr(instance, _OINFERENCE_DELEGATING_TRACKER, delegating_tracker)
    return delegating_tracker


def _ensure_client_hooks(instance: Any, delegating_tracker: _DelegatingToolSpanTracker) -> bool:
    if getattr(instance, _OINFERENCE_HOOKS_INJECTED, False):
        return True
    options = getattr(instance, "options", None)
    merged = _merge_hooks(options, delegating_tracker)
    if merged is None:
        return False
    try:
        setattr(instance, "options", merged)
        setattr(instance, _OINFERENCE_HOOKS_INJECTED, True)
    except Exception:
        return False
    return True


def _extract_message_content(message: Any) -> list[Any] | None:
    content = getattr(message, "content", None)
    if isinstance(content, list):
        return content
    inner = _get_field(message, "message")
    content = _get_field(inner, "content")
    return content if isinstance(content, list) else None


def _update_tool_spans_from_messages(
    message: Any,
    tool_tracker: _ToolSpanTracker,
    parent_tool_use_id: Any = None,
) -> None:
    try:
        content = _extract_message_content(message)
        if content is None:
            return

        for block in content:
            if _is_tool_use_block(block):
                tool_use_id = _get_field(block, "id", "")
                tool_name = _get_field(block, "name", "")
                tool_input = _get_field(block, "input", {})
                tool_tracker.start_tool_span(
                    tool_name,
                    tool_input,
                    tool_use_id,
                    parent_tool_use_id,
                )
            elif _is_tool_result_block(block):
                tool_use_id = _get_field(block, "tool_use_id", "")
                result_content = _get_field(block, "content")
                if _get_field(block, "is_error"):
                    tool_tracker.end_tool_span_with_error(tool_use_id, "Tool execution error")
                else:
                    tool_tracker.end_tool_span(tool_use_id, result_content)
    except Exception:
        pass


def _is_tool_use_block(block: Any) -> bool:
    block_type = _get_field(block, "type")
    if block_type and str(block_type) == "tool_use":
        return _get_field(block, "id") is not None
    return (
        _get_field(block, "id") is not None
        and _get_field(block, "name") is not None
        and _get_field(block, "input") is not None
        and _get_field(block, "tool_use_id") is None
    )


def _is_tool_result_block(block: Any) -> bool:
    return _get_field(block, "tool_use_id") is not None


def _is_text_block(block: Any) -> bool:
    block_type = _get_field(block, "type")
    if block_type:
        return str(block_type) == "text"
    # Duck-typing fallback: has 'text' but not 'id' or 'tool_use_id'
    return (
        _get_field(block, "text") is not None
        and _get_field(block, "id") is None
        and _get_field(block, "tool_use_id") is None
    )


def _get_output_message_attributes(message: Any, message_index: int) -> dict[str, Any]:
    """Extract llm.output_messages.* attributes from an assistant message."""
    attrs: dict[str, Any] = {}
    try:
        content = _extract_message_content(message)
        if content is None:
            return attrs
        tool_index = 0
        text_index = 0
        has_content = False
        for block in content:
            if _is_tool_use_block(block):
                has_content = True
                prefix = f"{LLM_OUTPUT_MESSAGES}.{message_index}.{MESSAGE_TOOL_CALLS}.{tool_index}"
                if tool_id := _get_field(block, "id"):
                    attrs[f"{prefix}.{TOOL_CALL_ID}"] = str(tool_id)
                if tool_name := _get_field(block, "name"):
                    attrs[f"{prefix}.{TOOL_CALL_FUNCTION_NAME}"] = str(tool_name)
                attrs[f"{prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"] = safe_json_dumps(
                    _get_field(block, "input") or {}
                )
                tool_index += 1
            elif _is_text_block(block):
                has_content = True
                if text := _get_field(block, "text"):
                    attrs[
                        f"{LLM_OUTPUT_MESSAGES}.{message_index}.{MESSAGE_CONTENT}.{text_index}"
                    ] = str(text)
                    text_index += 1
        if has_content:
            role = _get_field(message, "role") or "assistant"
            attrs[f"{LLM_OUTPUT_MESSAGES}.{message_index}.{MESSAGE_ROLE}"] = str(role)
    except Exception:
        pass
    return attrs


class _SubagentState:
    def __init__(self, span: trace_api.Span) -> None:
        self.span = span
        self.has_error = False


class _SubagentSpanTracker:
    def __init__(
        self,
        tracer: trace_api.Tracer,
        root_span: trace_api.Span | None,
        tool_tracker: _ToolSpanTracker,
    ) -> None:
        self._tracer = tracer
        self._root_span = root_span
        self._tool_tracker = tool_tracker
        self._in_flight: dict[str, _SubagentState] = {}

    def get_or_create(self, parent_tool_use_id: Any) -> _SubagentState:
        key = str(parent_tool_use_id)
        state = self._in_flight.get(key)
        if state is not None:
            return state

        parent_tool_span = self._tool_tracker.get_in_flight_span(parent_tool_use_id)
        tool_name = self._tool_tracker.get_tool_name(parent_tool_use_id)
        span_name = f"ClaudeAgentSDK.{tool_name}" if tool_name else "ClaudeAgentSDK.Subagent"
        attributes = {OPENINFERENCE_SPAN_KIND: AGENT}
        if tool_name:
            attributes[AGENT_NAME] = tool_name
        parent_span = parent_tool_span or self._root_span
        ctx = trace_api.set_span_in_context(parent_span) if parent_span is not None else None
        span = self._tracer.start_span(span_name, context=ctx, attributes=attributes)
        state = _SubagentState(span)
        self._in_flight[key] = state
        return state

    def process_message(self, message: Any) -> bool:
        parent_tool_use_id = _get_field(message, "parent_tool_use_id")
        if parent_tool_use_id is None:
            return False
        if str(parent_tool_use_id) == "":
            return False
        state = self.get_or_create(parent_tool_use_id)
        state.has_error = _process_message(message, state.span) or state.has_error
        if _is_result_success_message(message) or _is_result_error_message(message):
            self.end(parent_tool_use_id)
        return True

    def end(self, parent_tool_use_id: Any) -> None:
        key = str(parent_tool_use_id)
        state = self._in_flight.pop(key, None)
        if state is None:
            return
        try:
            if state.span.is_recording() and not state.has_error:
                state.span.set_status(trace_api.StatusCode.OK)
        finally:
            state.span.end()

    def end_all(self) -> None:
        for key in list(self._in_flight.keys()):
            self.end(key)


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

        prompt, options = _extract_prompt_and_options(args, kwargs)

        span = self._tracer.start_span(
            "ClaudeAgentSDK.query",
            attributes=dict(
                [
                    (OPENINFERENCE_SPAN_KIND, AGENT),
                    (LLM_SYSTEM, LLM_SYSTEM_ANTHROPIC),
                    *_format_prompt_attributes(prompt).items(),
                ]
                + list(get_attributes_from_context())
            ),
        )

        ctx = trace_api.set_span_in_context(span)
        token = context_api.attach(ctx)

        has_error = False
        subagent_tracker: _SubagentSpanTracker | None = None

        def _resolve_parent_span(parent_tool_use_id: Any) -> trace_api.Span | None:
            if subagent_tracker is None:
                return None
            return subagent_tracker.get_or_create(parent_tool_use_id).span

        tool_tracker = _ToolSpanTracker(
            self._tracer,
            span,
            parent_span_resolver=_resolve_parent_span,
        )
        subagent_tracker = _SubagentSpanTracker(self._tracer, span, tool_tracker)

        merged_options = _merge_hooks(options, tool_tracker)
        if merged_options is not None:
            args, kwargs = _apply_options(args, kwargs, merged_options)

        output_message_index = 0
        try:
            async for message in wrapped(*args, **kwargs):
                parent_tool_use_id = _get_field(message, "parent_tool_use_id")
                if subagent_tracker.process_message(message):
                    _update_tool_spans_from_messages(
                        message,
                        tool_tracker,
                        parent_tool_use_id=parent_tool_use_id,
                    )
                    yield message
                    continue
                has_error = _process_message(message, span) or has_error
                _update_tool_spans_from_messages(
                    message,
                    tool_tracker,
                    parent_tool_use_id=parent_tool_use_id,
                )
                msg_attrs = _get_output_message_attributes(message, output_message_index)
                if msg_attrs:
                    span.set_attributes(msg_attrs)
                    output_message_index += 1
                yield message
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            err_msg = f"{type(exc).__name__}: {exc}"
            span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, err_msg))
            has_error = True
            raise
        finally:
            tool_tracker.end_all_in_flight()
            subagent_tracker.end_all()
            try:
                if span.is_recording() and not has_error:
                    span.set_status(trace_api.StatusCode.OK)
            finally:
                span.end()
                context_api.detach(token)


class _ClientQueryWrapper:
    """Records prompt on the client for the next receive_response() span."""

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
        delegating_tracker = _get_or_create_delegating_tracker(instance)
        _ensure_client_hooks(instance, delegating_tracker)
        return await wrapped(*args, **kwargs)


class _ClientConnectWrapper:
    """Records initial prompt from connect() for the first receive_response() span."""

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
        delegating_tracker = _get_or_create_delegating_tracker(instance)
        _ensure_client_hooks(instance, delegating_tracker)
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

        span = self._tracer.start_span(
            "ClaudeAgentSDK.ClaudeSDKClient.receive_response",
            attributes=dict(
                [
                    (OPENINFERENCE_SPAN_KIND, AGENT),
                    (LLM_SYSTEM, LLM_SYSTEM_ANTHROPIC),
                    *_format_prompt_attributes(prompt).items(),
                ]
                + list(get_attributes_from_context())
            ),
        )
        ctx = trace_api.set_span_in_context(span)
        token = context_api.attach(ctx)

        has_error = False
        subagent_tracker: _SubagentSpanTracker | None = None

        def _resolve_parent_span(parent_tool_use_id: Any) -> trace_api.Span | None:
            if subagent_tracker is None:
                return None
            return subagent_tracker.get_or_create(parent_tool_use_id).span

        tool_tracker = _ToolSpanTracker(
            self._tracer,
            span,
            parent_span_resolver=_resolve_parent_span,
        )
        subagent_tracker = _SubagentSpanTracker(self._tracer, span, tool_tracker)

        delegating_tracker = _get_or_create_delegating_tracker(instance)

        hooks_injected = _ensure_client_hooks(instance, delegating_tracker)
        if hooks_injected:
            delegating_tracker.set_delegate(tool_tracker)

        output_message_index = 0
        try:
            async for message in wrapped(*args, **kwargs):
                parent_tool_use_id = _get_field(message, "parent_tool_use_id")
                if subagent_tracker.process_message(message):
                    _update_tool_spans_from_messages(
                        message,
                        tool_tracker,
                        parent_tool_use_id=parent_tool_use_id,
                    )
                    yield message
                    continue
                has_error = _process_message(message, span) or has_error
                _update_tool_spans_from_messages(
                    message,
                    tool_tracker,
                    parent_tool_use_id=parent_tool_use_id,
                )
                msg_attrs = _get_output_message_attributes(message, output_message_index)
                if msg_attrs:
                    span.set_attributes(msg_attrs)
                    output_message_index += 1
                yield message
        except Exception as exc:  # noqa: BLE001
            span.record_exception(exc)
            err_msg = f"{type(exc).__name__}: {exc}"
            span.set_status(trace_api.Status(trace_api.StatusCode.ERROR, err_msg))
            has_error = True
            raise
        finally:
            tool_tracker.end_all_in_flight()
            subagent_tracker.end_all()
            if hooks_injected:
                delegating_tracker.clear_delegate()
            try:
                if span.is_recording() and not has_error:
                    span.set_status(trace_api.StatusCode.OK)
            finally:
                span.end()
                context_api.detach(token)
