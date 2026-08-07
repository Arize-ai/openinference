from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractContextManager
from inspect import signature
from typing import Any

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    get_input_attributes,
    get_output_attributes,
    safe_json_dumps,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)


def _arguments(
    wrapped: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        return dict(signature(wrapped).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _io_attributes(
    value: Any, get_attributes: Callable[[Any], dict[str, AttributeValue]]
) -> dict[str, AttributeValue]:
    try:
        return get_attributes(value)
    except Exception:
        return get_attributes("<unserializable>")


def _agent_name(agent: Any) -> str:
    return str(getattr(agent, "name", None) or type(agent).__name__)


def _chat_output(result: Any) -> Any:
    history = getattr(result, "chat_history", None)
    if history:
        last = history[-1]
        return last.get("content") if isinstance(last, Mapping) else last
    return result


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
            **_io_attributes(input_value, get_input_attributes),
            **attributes,
        },
    )


def _finish_span(span: trace_api.Span, output: Any) -> None:
    span.set_attributes(_io_attributes(output, get_output_attributes))


def _record_exception(span: trace_api.Span, error: BaseException) -> None:
    span.set_status(trace_api.StatusCode.ERROR, str(error))
    span.record_exception(error)


def _use_span(span: trace_api.Span) -> AbstractContextManager[trace_api.Span]:
    # _record_exception handles escaping exceptions; use_span must not record
    # them too, or every error span carries duplicate exception events.
    return trace_api.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    )


class _ChatWrapper:
    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    def _span(self, instance: Any, method: str, bound: Mapping[str, Any]) -> trace_api.Span:
        recipient = bound.get("recipient")
        sender_name = _agent_name(instance)
        recipient_name = _agent_name(recipient) if recipient is not None else "unknown"
        return _start_span(
            self._tracer,
            f"{sender_name}.{method}",
            OpenInferenceSpanKindValues.AGENT,
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

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        span = self._span(instance, wrapped.__name__, _arguments(wrapped, args, kwargs))
        try:
            with _use_span(span):
                result = wrapped(*args, **kwargs)
            _finish_span(span, _chat_output(result))
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
        span = self._span(instance, wrapped.__name__, _arguments(wrapped, args, kwargs))
        try:
            with _use_span(span):
                result = await wrapped(*args, **kwargs)
            _finish_span(span, _chat_output(result))
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

    def _span(self, instance: Any, method: str, bound: Mapping[str, Any]) -> trace_api.Span:
        agent_name = _agent_name(instance)
        sender = bound.get("sender")
        return _start_span(
            self._tracer,
            f"{agent_name}.{method}",
            OpenInferenceSpanKindValues.AGENT,
            bound.get("messages"),
            {
                SpanAttributes.AGENT_NAME: agent_name,
                "ag2.sender.name": _agent_name(sender) if sender is not None else "unknown",
            },
        )

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        span = self._span(instance, wrapped.__name__, _arguments(wrapped, args, kwargs))
        try:
            with _use_span(span):
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
        span = self._span(instance, wrapped.__name__, _arguments(wrapped, args, kwargs))
        try:
            with _use_span(span):
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
    def _normalized_call(
        args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        # execute_function requires a mapping, but callers historically passed
        # bare function names as strings; normalize before delegating.
        if isinstance(kwargs.get("func_call"), str):
            kwargs = {**kwargs, "func_call": {"name": kwargs["func_call"]}}
        elif args and isinstance(args[0], str):
            args = ({"name": args[0]}, *args[1:])
        return args, kwargs

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

    def _span(self, instance: Any, bound: Mapping[str, Any]) -> trace_api.Span:
        func_call = bound.get("func_call")
        name, attributes = self._attributes(instance, func_call, bound.get("call_id"))
        return _start_span(
            self._tracer, name, OpenInferenceSpanKindValues.TOOL, func_call, attributes
        )

    @staticmethod
    def _set_result_status(span: trace_api.Span, result: Any) -> None:
        if isinstance(result, tuple) and result and result[0] is False:
            span.set_status(trace_api.StatusCode.ERROR, "tool execution failed")
        else:
            span.set_status(trace_api.StatusCode.OK)

    def __call__(
        self,
        wrapped: Callable[..., Any],
        instance: Any,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> Any:
        if context_api.get_value(context_api._SUPPRESS_INSTRUMENTATION_KEY):
            return wrapped(*args, **kwargs)
        args, kwargs = self._normalized_call(args, kwargs)
        span = self._span(instance, _arguments(wrapped, args, kwargs))
        try:
            with _use_span(span):
                result = wrapped(*args, **kwargs)
            _finish_span(span, result)
            self._set_result_status(span, result)
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
        args, kwargs = self._normalized_call(args, kwargs)
        span = self._span(instance, _arguments(wrapped, args, kwargs))
        try:
            with _use_span(span):
                result = await wrapped(*args, **kwargs)
            _finish_span(span, result)
            self._set_result_status(span, result)
            return result
        except Exception as error:
            _record_exception(span, error)
            raise
        finally:
            span.end()
