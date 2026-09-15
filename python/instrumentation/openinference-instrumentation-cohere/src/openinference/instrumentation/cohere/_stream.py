import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.cohere._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.cohere._with_span import _WithSpan
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ("_Stream",)


class _EventAccumulator:
    """Accumulates ``chat_stream`` events into a chat-response-shaped object.

    Cohere emits the assistant turn as a sequence of typed events rather than as a
    single payload, so the pieces are collected here and materialized into the same
    shape ``_ResponseAttributesExtractor`` reads from a non-streaming response.
    """

    def __init__(self) -> None:
        self._texts: List[str] = []
        self._tool_plans: List[str] = []
        self._usage: Any = None
        self._finish_reason: Optional[str] = None
        # Keyed by the event's ``index`` so parallel tool calls stay separate.
        self._tool_calls: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"id": None, "name": None, "arguments": []}
        )
        self._response: Optional[SimpleNamespace] = None

    def process_event(self, event: Any) -> None:
        try:
            self._process_event(event)
        except Exception:
            logger.exception(f"Failed to process stream event of type {type(event)}")

    def _process_event(self, event: Any) -> None:
        event_type = getattr(event, "type", None)
        delta = getattr(event, "delta", None)
        message = getattr(delta, "message", None)
        if event_type == "content-delta":
            if (content := getattr(message, "content", None)) is not None:
                if text := getattr(content, "text", None):
                    self._texts.append(text)
        elif event_type == "tool-plan-delta":
            if tool_plan := getattr(message, "tool_plan", None):
                self._tool_plans.append(tool_plan)
        elif event_type in ("tool-call-start", "tool-call-delta"):
            if (tool_call := getattr(message, "tool_calls", None)) is None:
                return
            entry = self._tool_calls[getattr(event, "index", None) or 0]
            if tool_call_id := getattr(tool_call, "id", None):
                entry["id"] = tool_call_id
            if function := getattr(tool_call, "function", None):
                if name := getattr(function, "name", None):
                    entry["name"] = name
                if arguments := getattr(function, "arguments", None):
                    entry["arguments"].append(arguments)
        elif event_type == "message-end":
            if usage := getattr(delta, "usage", None):
                self._usage = usage
            if finish_reason := getattr(delta, "finish_reason", None):
                self._finish_reason = finish_reason

    def _get_response(self) -> SimpleNamespace:
        if self._response is not None:
            return self._response
        tool_calls = [
            SimpleNamespace(
                id=entry["id"],
                type="function",
                function=SimpleNamespace(
                    name=entry["name"],
                    arguments="".join(entry["arguments"]),
                ),
            )
            for _, entry in sorted(self._tool_calls.items())
        ]
        self._response = SimpleNamespace(
            message=SimpleNamespace(
                role="assistant",
                content="".join(self._texts),
                tool_plan="".join(self._tool_plans) or None,
                tool_calls=tool_calls or None,
            ),
            usage=self._usage,
            finish_reason=self._finish_reason,
        )
        return self._response

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        message = self._get_response().message
        if message.content and not message.tool_calls:
            yield SpanAttributes.OUTPUT_VALUE, message.content
            return
        payload: Dict[str, Any] = {"role": message.role}
        if message.content:
            payload["content"] = message.content
        if message.tool_plan:
            payload["tool_plan"] = message.tool_plan
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in message.tool_calls
            ]
        yield SpanAttributes.OUTPUT_VALUE, safe_json_dumps(payload)
        yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value

    def get_extra_attributes(
        self,
        response_extractor: _ResponseAttributesExtractor,
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from response_extractor.get_extra_attributes(response=self._get_response())


class _Stream(ObjectProxy):  # type: ignore[misc,type-arg,unused-ignore]
    """Wraps a ``chat_stream`` iterator so the span ends when the stream is consumed.

    The span must not be finished when ``chat_stream`` returns, because none of the
    response has arrived yet; it is finished once iteration completes, errors, or the
    stream's context manager exits.
    """

    __slots__ = ("_self_with_span", "_self_accumulator", "_self_response_extractor")

    def __init__(
        self,
        stream: Any,
        with_span: _WithSpan,
        response_extractor: _ResponseAttributesExtractor,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_accumulator = _EventAccumulator()
        self._self_response_extractor = response_extractor

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            event = self.__wrapped__.__next__()
        except StopIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_error(exception)
            raise
        self._self_accumulator.process_event(event)
        return event

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            event = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_error(exception)
            raise
        self._self_accumulator.process_event(event)
        return event

    def __enter__(self) -> "_Stream":
        self.__wrapped__.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        result = self.__wrapped__.__exit__(*args)
        self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
        return result

    async def __aenter__(self) -> "_Stream":
        await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        result = await self.__wrapped__.__aexit__(*args)
        self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
        return result

    def _finish_error(self, exception: BaseException) -> None:
        self._self_with_span.record_exception(exception)
        self._finish(
            trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def _finish(self, status: trace_api.Status) -> None:
        if self._self_with_span.is_finished:
            return
        attributes: Optional[Dict[str, AttributeValue]] = None
        try:
            attributes = dict(self._self_accumulator.get_attributes())
        except Exception:
            logger.exception("Failed to get attributes from stream")
        extra_attributes: Optional[Dict[str, AttributeValue]] = None
        try:
            extra_attributes = dict(
                self._self_accumulator.get_extra_attributes(self._self_response_extractor)
            )
        except Exception:
            logger.exception("Failed to get extra attributes from stream")
        self._self_with_span.finish_tracing(
            status=status,
            attributes=attributes,
            extra_attributes=extra_attributes,
        )
