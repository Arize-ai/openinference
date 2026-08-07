"""Wrappers that turn AG2 chats, replies, and tool calls into OpenInference spans."""

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
    """Bind a call's arguments to their parameter names, falling back to the keywords."""
    try:
        return dict(signature(wrapped).bind_partial(*args, **kwargs).arguments)
    except (TypeError, ValueError):
        return dict(kwargs)


def _io_attributes(
    value: Any, get_attributes: Callable[[Any], dict[str, AttributeValue]]
) -> dict[str, AttributeValue]:
    """Build input or output attributes, substituting a placeholder if serialization fails."""
    try:
        return get_attributes(value)
    except Exception:
        return get_attributes("<unserializable>")


def _agent_name(agent: Any) -> str:
    """Name an agent, falling back to its class when it is unnamed."""
    return str(getattr(agent, "name", None) or type(agent).__name__)


def _chat_output(result: Any) -> Any:
    """Reduce a ``ChatResult`` to its final message, which is the answer callers expect."""
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
    """Start a span carrying its kind, input value, and the caller's attributes."""
    return tracer.start_span(
        name,
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: kind.value,
            **_io_attributes(input_value, get_input_attributes),
            **attributes,
        },
    )


def _finish_span(span: trace_api.Span, output: Any) -> None:
    """Record the call's output on the span."""
    span.set_attributes(_io_attributes(output, get_output_attributes))


def _record_exception(span: trace_api.Span, error: BaseException) -> None:
    """Mark the span as failed and attach the exception."""
    span.set_status(trace_api.StatusCode.ERROR, str(error))
    span.record_exception(error)


def _use_span(span: trace_api.Span) -> AbstractContextManager[trace_api.Span]:
    """Make the span current without letting it record exceptions.

    ``_record_exception`` already handles anything that escapes, so leaving the
    defaults on would put a duplicate exception event on every error span.
    """
    return trace_api.use_span(
        span, end_on_exit=False, record_exception=False, set_status_on_exception=False
    )


class _ChatWrapper:
    """Traces ``initiate_chat`` and ``a_initiate_chat`` as the AGENT span for a chat."""

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
    """Traces ``generate_reply`` and ``a_generate_reply`` as an AGENT span per reply."""

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
    """Traces ``execute_function`` and ``a_execute_function`` as a TOOL span per call."""

    def __init__(self, tracer: trace_api.Tracer) -> None:
        self._tracer = tracer

    @staticmethod
    def _normalized_call(
        args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> tuple[tuple[Any, ...], Mapping[str, Any]]:
        """Promote a bare function name to the mapping ``execute_function`` requires.

        Callers historically passed the name as a string, so normalize before
        delegating rather than letting the call fail inside AG2.
        """
        if isinstance(kwargs.get("func_call"), str):
            kwargs = {**kwargs, "func_call": {"name": kwargs["func_call"]}}
        elif args and isinstance(args[0], str):
            args = ({"name": args[0]}, *args[1:])
        return args, kwargs

    @staticmethod
    def _attributes(agent: Any, func_call: Any, call_id: Any) -> tuple[str, dict[str, Any]]:
        """Return the tool's name and its span attributes, including parameter types."""
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
        """Fail the span when AG2 reports the execution flag as False."""
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
