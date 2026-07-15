from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, is_dataclass
from inspect import signature
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api

from openinference.instrumentation import (
    get_attributes_from_context,
    get_input_attributes,
    get_output_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

JSON = OpenInferenceMimeTypeValues.JSON


def _arguments(
    wrapped: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return dict(signature(wrapped).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _json_value(value: Any) -> str:
    try:
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)
        elif callable(model_dump := getattr(value, "model_dump", None)):
            value = model_dump(mode="json")
        return safe_json_dumps(value)
    except Exception:
        return safe_json_dumps("<unserializable>")


def _agent_name(agent: Any) -> str:
    return str(getattr(agent, "name", None) or type(agent).__name__)


def _start_span(
    tracer: trace_api.Tracer,
    name: str,
    kind: OpenInferenceSpanKindValues,
    input_value: Any,
    attributes: Mapping[str, Any],
) -> trace_api.Span:
    return tracer.start_span(
        name,
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: kind.value,
            **get_input_attributes(_json_value(input_value), mime_type=JSON),
            **attributes,
            **dict(get_attributes_from_context()),
        },
    )


def _finish_span(span: trace_api.Span, output: Any) -> None:
    span.set_attributes(get_output_attributes(_json_value(output), mime_type=JSON))


def _record_exception(span: trace_api.Span, error: BaseException) -> None:
    span.set_status(trace_api.StatusCode.ERROR, str(error))
    span.record_exception(error)


class _ChatWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        bound = _arguments(wrapped, args, kwargs)
        recipient = bound.get("recipient")
        sender_name = _agent_name(instance)
        recipient_name = _agent_name(recipient) if recipient is not None else "unknown"
        span = _start_span(
            self._tracer,
            f"{sender_name}.initiate_chat",
            OpenInferenceSpanKindValues.CHAIN,
            {
                "message": bound.get("message"),
                "sender": sender_name,
                "recipient": recipient_name,
            },
            {
                SpanAttributes.AGENT_NAME: sender_name,
                "ag2.recipient.name": recipient_name,
            },
        )
        try:
            with trace_api.use_span(span, end_on_exit=False):
                result = wrapped(*args, **kwargs)
            _finish_span(span, result)
            span.set_status(trace_api.StatusCode.OK)
            return result
        except Exception as error:
            _record_exception(span, error)
            raise
        finally:
            span.end()

    async def async_call(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        bound = _arguments(wrapped, args, kwargs)
        recipient = bound.get("recipient")
        sender_name = _agent_name(instance)
        recipient_name = _agent_name(recipient) if recipient is not None else "unknown"
        span = _start_span(
            self._tracer,
            f"{sender_name}.a_initiate_chat",
            OpenInferenceSpanKindValues.CHAIN,
            {
                "message": bound.get("message"),
                "sender": sender_name,
                "recipient": recipient_name,
            },
            {
                SpanAttributes.AGENT_NAME: sender_name,
                "ag2.recipient.name": recipient_name,
            },
        )
        try:
            with trace_api.use_span(span, end_on_exit=False):
                result = await wrapped(*args, **kwargs)
            _finish_span(span, result)
            span.set_status(trace_api.StatusCode.OK)
            return result
        except Exception as error:
            _record_exception(span, error)
            raise
        finally:
            span.end()


class _ReplyWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        bound = _arguments(wrapped, args, kwargs)
        agent_name = _agent_name(instance)
        sender = bound.get("sender")
        span = _start_span(
            self._tracer,
            f"{agent_name}.generate_reply",
            OpenInferenceSpanKindValues.AGENT,
            bound.get("messages"),
            {
                SpanAttributes.AGENT_NAME: agent_name,
                "ag2.sender.name": _agent_name(sender) if sender is not None else "unknown",
            },
        )
        try:
            with trace_api.use_span(span, end_on_exit=False):
                result = wrapped(*args, **kwargs)
            _finish_span(span, result)
            span.set_status(trace_api.StatusCode.OK)
            return result
        except Exception as error:
            _record_exception(span, error)
            raise
        finally:
            span.end()

    async def async_call(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        bound = _arguments(wrapped, args, kwargs)
        agent_name = _agent_name(instance)
        sender = bound.get("sender")
        span = _start_span(
            self._tracer,
            f"{agent_name}.a_generate_reply",
            OpenInferenceSpanKindValues.AGENT,
            bound.get("messages"),
            {
                SpanAttributes.AGENT_NAME: agent_name,
                "ag2.sender.name": _agent_name(sender) if sender is not None else "unknown",
            },
        )
        try:
            with trace_api.use_span(span, end_on_exit=False):
                result = await wrapped(*args, **kwargs)
            _finish_span(span, result)
            span.set_status(trace_api.StatusCode.OK)
            return result
        except Exception as error:
            _record_exception(span, error)
            raise
        finally:
            span.end()


class _ToolWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    @staticmethod
    def _attributes(agent: Any, func_call: Any, call_id: Any) -> tuple[str, dict[str, Any]]:
        call = func_call if isinstance(func_call, Mapping) else {}
        name = str(call.get("name") or "unknown")
        raw_arguments = call.get("arguments", "{}")
        if isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = raw_arguments
        else:
            arguments = raw_arguments
        attributes: dict[str, Any] = {
            SpanAttributes.TOOL_NAME: name,
            ToolCallAttributes.TOOL_CALL_FUNCTION_NAME: name,
            ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON: safe_json_dumps(arguments),
            SpanAttributes.AGENT_NAME: _agent_name(agent),
        }
        if call_id:
            attributes[ToolCallAttributes.TOOL_CALL_ID] = str(call_id)
        function = getattr(agent, "_function_map", {}).get(name)
        parameters = {
            parameter: getattr(annotation, "__name__", str(annotation))
            for parameter, annotation in getattr(function, "__annotations__", {}).items()
            if parameter != "return"
        }
        if parameters:
            attributes[SpanAttributes.TOOL_PARAMETERS] = safe_json_dumps(parameters)
        return name, attributes

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        bound = _arguments(wrapped, args, kwargs)
        func_call = bound.get("func_call")
        name, attributes = self._attributes(instance, func_call, bound.get("call_id"))
        span = _start_span(
            self._tracer,
            name,
            OpenInferenceSpanKindValues.TOOL,
            func_call,
            attributes,
        )
        try:
            with trace_api.use_span(span, end_on_exit=False):
                result = wrapped(*args, **kwargs)
            _finish_span(span, result)
            if isinstance(result, tuple) and result and result[0] is False:
                span.set_status(trace_api.StatusCode.ERROR, "tool execution failed")
            else:
                span.set_status(trace_api.StatusCode.OK)
            return result
        except Exception as error:
            _record_exception(span, error)
            raise
        finally:
            span.end()

    async def async_call(
        self,
        wrapped: Callable[..., Awaitable[Any]],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return await wrapped(*args, **kwargs)
        bound = _arguments(wrapped, args, kwargs)
        func_call = bound.get("func_call")
        name, attributes = self._attributes(instance, func_call, bound.get("call_id"))
        span = _start_span(
            self._tracer,
            name,
            OpenInferenceSpanKindValues.TOOL,
            func_call,
            attributes,
        )
        try:
            with trace_api.use_span(span, end_on_exit=False):
                result = await wrapped(*args, **kwargs)
            _finish_span(span, result)
            if isinstance(result, tuple) and result and result[0] is False:
                span.set_status(trace_api.StatusCode.ERROR, "tool execution failed")
            else:
                span.set_status(trace_api.StatusCode.OK)
            return result
        except Exception as error:
            _record_exception(span, error)
            raise
        finally:
            span.end()
