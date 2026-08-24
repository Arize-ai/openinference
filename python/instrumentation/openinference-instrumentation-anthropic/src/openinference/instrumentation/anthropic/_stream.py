from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Iterator,
    Optional,
    Tuple,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.anthropic._utils import (
    _finish_tracing,
    _get_token_counts,
)
from openinference.instrumentation.anthropic._with_span import _WithSpan
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from anthropic import Stream
    from anthropic.types import RawMessageStreamEvent


class _RawStreamInterceptor(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    """
    Wraps the raw HTTP stream inside a MessageStream. Forwards every event
    unchanged so MessageStream can run its own accumulation (accumulate_event),
    and calls _finish_tracing once the stream is exhausted or an error occurs.
    No custom accumulation is needed here because MessageStream.current_message_snapshot
    gives us the complete ParsedMessage at the end.
    """

    __slots__ = ("_self_with_span", "_self_message_stream")

    def __init__(
        self,
        raw_stream: "Stream[RawMessageStreamEvent]",
        with_span: "_WithSpan",
        message_stream: Any = None,
    ) -> None:
        super().__init__(raw_stream)
        self._self_with_span = with_span
        self._self_message_stream = message_stream

    def __iter__(self) -> Iterator["RawMessageStreamEvent"]:
        try:
            for item in self.__wrapped__:
                yield item
        except Exception as exception:
            self._self_with_span.record_exception(exception)
            self._finish_tracing(
                status=trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
            )
            raise
        self._finish_tracing(status=trace_api.Status(status_code=trace_api.StatusCode.OK))

    async def __aiter__(self) -> AsyncIterator["RawMessageStreamEvent"]:
        try:
            async for item in self.__wrapped__:
                yield item
        except Exception as exception:
            self._self_with_span.record_exception(exception)
            self._finish_tracing(
                status=trace_api.Status(
                    status_code=trace_api.StatusCode.ERROR,
                    description=f"{type(exception).__name__}: {exception}",
                )
            )
            raise
        self._finish_tracing(status=trace_api.Status(status_code=trace_api.StatusCode.OK))

    def _finish_tracing(self, status: Optional[trace_api.Status] = None) -> None:
        snapshot = None
        if self._self_message_stream is not None:
            try:
                snapshot = self._self_message_stream.current_message_snapshot
            except Exception:
                pass
        _finish_tracing(
            with_span=self._self_with_span,
            has_attributes=_MessageExtractor(snapshot),
            status=status,
        )


class _MessagesStream(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = (
        "_response_accumulator",
        "_with_span",
    )

    def __init__(
        self,
        stream: "Stream[RawMessageStreamEvent]",
        with_span: _WithSpan,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _MessageResponseAccumulator()
        self._with_span = with_span

    def __iter__(self) -> Iterator["RawMessageStreamEvent"]:
        try:
            for item in self.__wrapped__:
                self._response_accumulator.process_chunk(item)
                yield item
        except Exception as exception:
            status = trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
            self._with_span.record_exception(exception)
            self._finish_tracing(status=status)
            raise
        # completed without exception
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    async def __aiter__(self) -> AsyncIterator["RawMessageStreamEvent"]:
        try:
            async for item in self.__wrapped__:
                self._response_accumulator.process_chunk(item)
                yield item
        except Exception as exception:
            status = trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
            self._with_span.record_exception(exception)
            self._finish_tracing(status=status)
            raise
        # completed without exception
        status = trace_api.Status(
            status_code=trace_api.StatusCode.OK,
        )
        self._finish_tracing(status=status)

    def _finish_tracing(
        self,
        status: Optional[trace_api.Status] = None,
    ) -> None:
        _finish_tracing(
            with_span=self._with_span,
            has_attributes=_MessageExtractor(self._response_accumulator._result()),
            status=status,
        )


class _MessageResponseAccumulator:
    """Accumulates raw SSE events into a ParsedMessage using the SDK's own accumulate_event."""

    __slots__ = ("_snapshot",)

    def __init__(self) -> None:
        self._snapshot: Any = None

    def process_chunk(self, chunk: "RawMessageStreamEvent") -> None:
        from anthropic.lib.streaming._messages import accumulate_event

        try:
            self._snapshot = accumulate_event(event=chunk, current_snapshot=self._snapshot)
        except Exception:
            pass

    def _result(self) -> Any:
        return self._snapshot


class _MessageExtractor:
    """
    Extracts span attributes from a ParsedMessage (or Message) snapshot.
    Used by both the messages.stream() path (via current_message_snapshot)
    and the messages.create(stream=True) path (via _MessageResponseAccumulator).
    """

    __slots__ = ("_snapshot",)

    def __init__(self, snapshot: Any) -> None:
        self._snapshot = snapshot

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        snapshot = self._snapshot
        if snapshot is None:
            return
        yield SpanAttributes.OUTPUT_VALUE, snapshot.model_dump_json()
        yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
        yield (
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}",
            snapshot.role,
        )
        if stop_reason := getattr(snapshot, "stop_reason", None):
            yield SpanAttributes.LLM_FINISH_REASON, stop_reason
        tool_idx = 0
        for block_idx, block in enumerate(snapshot.content):
            content_prefix = (
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
                f"{MessageAttributes.MESSAGE_CONTENTS}.{block_idx}"
            )
            if block.type == "text":
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "text",
                )
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                    block.text,
                )
            elif block.type == "thinking":
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "reasoning",
                )
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                    block.thinking,
                )
                if signature := block.signature:
                    yield (
                        f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}",
                        signature,
                    )
            elif block.type == "redacted_thinking":
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "reasoning",
                )
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_DATA}",
                    block.data,
                )
            elif block.type == "tool_use":
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_ID}",
                    block.id,
                )
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    block.name,
                )
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    safe_json_dumps(block.input),
                )
                yield (
                    f"{content_prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "tool_use",
                )
                yield (
                    f"{content_prefix}.{ToolCallAttributes.TOOL_CALL_ID}",
                    block.id,
                )
                yield (
                    f"{content_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    block.name,
                )
                yield (
                    f"{content_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    safe_json_dumps(block.input),
                )
                tool_idx += 1
        yield from _get_token_counts(snapshot.usage)
