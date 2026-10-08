from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    Optional,
    Tuple,
    Type,
)

from opentelemetry import trace as trace_api
from wrapt import ObjectProxy

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.anthropic._types import AttributeValue
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
    from httpx2 import Headers

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

    __slots__ = ("_self_with_span", "_self_message_stream", "_self_is_exhausted")

    def __init__(
        self,
        raw_stream: "Stream[RawMessageStreamEvent]",
        with_span: "_WithSpan",
        message_stream: Any = None,
    ) -> None:
        super().__init__(raw_stream)
        self._self_with_span = with_span
        self._self_message_stream = message_stream
        # Whether iteration ran to the end of the stream. Leaving the stream early closes this
        # generator instead, so the SDK snapshot can still hold in-progress values.
        self._self_is_exhausted = False

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
        self._self_is_exhausted = True
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
        self._self_is_exhausted = True
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
            has_attributes=_MessageExtractor(snapshot, is_exhausted=self._self_is_exhausted),
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
        *,
        is_beta: bool = False,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _MessageResponseAccumulator(
            is_beta=is_beta,
            request_headers=stream.response.request.headers,
        )
        self._with_span = with_span

    # The SDK stream's context manager returns the SDK stream, which would bypass the iteration
    # below, so these return the proxy. Exiting finishes the span if iteration has not, e.g. when
    # the stream is left early, recording the exception that ended the context, e.g. a
    # CancelledError, which iteration does not catch.

    def __enter__(self) -> "_MessagesStream":
        self.__wrapped__.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        try:
            self.__wrapped__.__exit__(exc_type, exc_val, exc_tb)
        except BaseException as exception:
            # e.g. closing the response failed
            self._finish_tracing_on_exit(exception)
            raise
        self._finish_tracing_on_exit(exc_val)

    async def __aenter__(self) -> "_MessagesStream":
        await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        try:
            await self.__wrapped__.__aexit__(exc_type, exc_val, exc_tb)
        except BaseException as exception:
            # e.g. closing the response failed, or the task was cancelled while it closed
            self._finish_tracing_on_exit(exception)
            raise
        self._finish_tracing_on_exit(exc_val)

    def _finish_tracing_on_exit(self, exception: Optional[BaseException]) -> None:
        # GeneratorExit: a generator holding the context was closed, which leaves the stream
        # early rather than failing the request
        if exception is None or isinstance(exception, GeneratorExit):
            self._finish_tracing()
            return
        self._with_span.record_exception(exception)
        self._finish_tracing(
            status=trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

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

    __slots__ = ("_is_beta", "_request_headers", "_snapshot", "_json_bufs")

    def __init__(
        self,
        *,
        is_beta: bool,
        request_headers: "Headers",
    ) -> None:
        self._is_beta = is_beta
        self._request_headers = request_headers
        self._snapshot: Any = None
        # Buffers partial tool-use input JSON across events, keyed by content block
        # index.
        self._json_bufs: Dict[int, bytes] = {}

    def process_chunk(self, chunk: "RawMessageStreamEvent") -> None:
        # Beta and stable chunks need their matching accumulate_event; beta's
        # raises on stable chunks and vice versa silently drops updates.
        if self._is_beta:
            from anthropic.lib.streaming._beta_messages import (
                accumulate_event as accumulate_beta_event,
            )

            beta_kwargs: Dict[str, Any] = dict(
                event=chunk,
                current_snapshot=self._snapshot,
                request_headers=self._request_headers,
                json_bufs=self._json_bufs,
            )
            try:
                self._snapshot = accumulate_beta_event(**beta_kwargs)
            except Exception:
                pass
        else:
            from anthropic.lib.streaming._messages import accumulate_event

            try:
                self._snapshot = accumulate_event(
                    event=chunk,
                    current_snapshot=self._snapshot,
                    json_bufs=self._json_bufs,
                )
            except Exception:
                pass

    def _result(self) -> Any:
        return self._snapshot


class _MessageExtractor:
    """
    Extracts span attributes from a ParsedMessage (or Message) snapshot.
    Used by both the messages.stream() path (via current_message_snapshot)
    and the messages.create(stream=True) path (via _MessageResponseAccumulator).

    ``is_exhausted`` says whether the stream was read to its end. A stream left early
    leaves tool input at the empty placeholder the SDK sends in content_block_start,
    because the arguments are only parsed into the snapshot once the block completes, so
    that placeholder is not a final value and must not be recorded as one.
    """

    __slots__ = ("_snapshot", "_is_exhausted")

    def __init__(self, snapshot: Any, is_exhausted: bool = True) -> None:
        self._snapshot = snapshot
        self._is_exhausted = is_exhausted

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        snapshot = self._snapshot
        if snapshot is None:
            return
        yield SpanAttributes.OUTPUT_VALUE, snapshot.model_dump_json()
        yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
        if model_name := getattr(snapshot, "model", None):
            yield SpanAttributes.LLM_MODEL_NAME, model_name
            yield SpanAttributes.LLM_RESPONSE_MODEL_NAME, model_name
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
                # An unfinished stream can leave the tool input at the SDK's empty
                # placeholder, which the model never sent as final arguments (#3904).
                arguments = self._tool_arguments(block)
                if arguments is not None:
                    yield (
                        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
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
                if arguments is not None:
                    yield (
                        f"{content_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
                    )
                tool_idx += 1
        yield from _get_token_counts(snapshot.usage)

    def _tool_arguments(self, block: Any) -> Optional[str]:
        """
        Returns the serialized tool arguments, or None if they are an unfinished stream's
        placeholder rather than a value the model produced.
        """
        if not self._is_exhausted and not getattr(block, "input", None):
            return None
        return safe_json_dumps(block.input)
