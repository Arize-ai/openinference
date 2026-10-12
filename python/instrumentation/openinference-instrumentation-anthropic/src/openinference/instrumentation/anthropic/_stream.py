import inspect
import json
import logging
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Dict,
    Iterator,
    Optional,
    Set,
    Tuple,
    Type,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.anthropic._content import (
    TOOL_CALL_BLOCK_TYPES,
    _attributes_or_nothing,
    _get_metadata_attributes,
    _get_output_message_attributes,
)
from openinference.instrumentation.anthropic._utils import (
    _finish_tracing,
    _get_token_counts,
)
from openinference.instrumentation.anthropic._with_span import _WithSpan
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)

if TYPE_CHECKING:
    from httpx2 import Headers

    from anthropic import Stream
    from anthropic.types import RawMessageStreamEvent


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
            has_attributes=_MessageExtractor(
                snapshot,
                context_metadata=self._self_with_span.context_metadata,
                config=self._self_with_span.config,
                request_messages=self._self_with_span.request_messages,
            ),
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

    # Closing the stream, or dropping it, before it is exhausted never ends iteration, so these
    # finish the span with what was read so far, leaving the status unset as when the context is
    # left early.

    def close(self) -> Any:
        try:
            result = self.__wrapped__.close()
        except BaseException as exception:
            self._finish_tracing_on_exit(exception)
            raise
        if inspect.isawaitable(result):
            # the async SDK stream closes in a coroutine
            return self._finish_tracing_after_close(result)
        self._finish_tracing_on_exit(None)
        return result

    async def _finish_tracing_after_close(self, closing: Awaitable[Any]) -> Any:
        try:
            result = await closing
        except BaseException as exception:
            self._finish_tracing_on_exit(exception)
            raise
        self._finish_tracing_on_exit(None)
        return result

    def __del__(self) -> None:
        try:
            self._finish_tracing()
        except BaseException:
            pass

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
        except GeneratorExit:
            # the loop was left early, and the stream can still be read again, so the span is
            # finished by close, exiting the context, or dropping the stream
            raise
        except BaseException as exception:
            # e.g. a KeyboardInterrupt or a cancelled task while waiting for the next event
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
        except GeneratorExit:
            raise
        except BaseException as exception:
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
            has_attributes=_MessageExtractor(
                self._response_accumulator._result(),
                context_metadata=self._with_span.context_metadata,
                config=self._with_span.config,
                request_messages=self._with_span.request_messages,
            ),
            status=status,
        )


class _MessageResponseAccumulator:
    """Accumulates raw SSE events into a ParsedMessage using the SDK's own accumulate_event."""

    __slots__ = ("_is_beta", "_request_headers", "_snapshot", "_json_bufs", "_stale_inputs")

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
        # Indices of content blocks whose tool-use input could not be updated from a
        # delta, so their input must be parsed again from the buffer at the end.
        self._stale_inputs: Set[int] = set()

    def process_chunk(self, chunk: "RawMessageStreamEvent") -> None:
        # Beta and stable chunks need their matching accumulate_event; beta's
        # raises on stable chunks and vice versa silently drops updates.
        try:
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
                self._snapshot = accumulate_beta_event(**beta_kwargs)
            else:
                from anthropic.lib.streaming._messages import accumulate_event

                self._snapshot = accumulate_event(
                    event=chunk,
                    current_snapshot=self._snapshot,
                    json_bufs=self._json_bufs,
                )
        except Exception as exception:
            # Keep going with the rest of the stream. An input_json_delta that leaves the partial
            # JSON unparseable is not added to the buffer by the SDK, so it is added here, or the
            # next delta would be appended to a buffer missing this one and the tool call's
            # arguments would be silently wrong.
            logger.debug("Failed to accumulate a stream event", exc_info=True)
            self._buffer_failed_input_json_delta(chunk, exception)

    def _buffer_failed_input_json_delta(self, chunk: Any, exception: Exception) -> None:
        """
        Buffers the delta only if the SDK failed to parse the JSON, which it raises as a
        ValueError, for a block at the delta's index that tracks tool input. Any other failure
        may have happened before the delta was matched to its block, so buffering it could put
        the arguments on the wrong block.
        """
        try:
            if not isinstance(exception, ValueError):
                return
            if getattr(chunk, "type", None) != "content_block_delta":
                return
            delta = chunk.delta
            if getattr(delta, "type", None) != "input_json_delta":
                return
            index = chunk.index
            content = getattr(self._snapshot, "content", None) or []
            if not 0 <= index < len(content):
                return
            if getattr(content[index], "type", None) not in TOOL_CALL_BLOCK_TYPES:
                return
            self._json_bufs[index] = self._json_bufs.get(index, b"") + bytes(
                delta.partial_json, "utf-8"
            )
            self._stale_inputs.add(index)
        except Exception:
            logger.debug("Failed to buffer a tool-use input delta", exc_info=True)

    def _result(self) -> Any:
        snapshot = self._snapshot
        for index in self._stale_inputs:
            # The arguments are recorded only if the complete JSON parses, otherwise the last
            # partial parse is kept.
            try:
                snapshot.content[index].input = json.loads(self._json_bufs[index])
            except Exception:
                logger.debug("Failed to parse tool-use input JSON", exc_info=True)
        self._stale_inputs.clear()
        return snapshot


class _MessageExtractor:
    """
    Extracts span attributes from a ParsedMessage (or Message) snapshot.
    Used by both the messages.stream() path (via current_message_snapshot)
    and the messages.create(stream=True) path (via _MessageResponseAccumulator).
    """

    __slots__ = ("_snapshot", "_context_metadata", "_config", "_request_messages")

    def __init__(
        self,
        snapshot: Any,
        context_metadata: Optional[AttributeValue] = None,
        config: Optional[TraceConfig] = None,
        request_messages: Any = None,
    ) -> None:
        self._snapshot = snapshot
        self._context_metadata = context_metadata
        self._config = config
        self._request_messages = request_messages

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        snapshot = self._snapshot
        if snapshot is None:
            return
        # by_alias, so the fields have the API's names, e.g. a fallback block's `from`
        yield SpanAttributes.OUTPUT_VALUE, snapshot.model_dump_json(by_alias=True)
        yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value
        if model_name := getattr(snapshot, "model", None):
            yield SpanAttributes.LLM_MODEL_NAME, model_name
            yield SpanAttributes.LLM_RESPONSE_MODEL_NAME, model_name
        yield from _attributes_or_nothing(_get_output_message_attributes, snapshot)
        yield from _attributes_or_nothing(
            _get_metadata_attributes,
            snapshot,
            self._context_metadata,
            self._config,
            self._request_messages,
        )
        # Token counts come from a usage snapshot that can be partial or
        # delta-shaped while streaming. Contain any failure here (the same
        # @_stop_on_exception treatment the non-streaming path gives
        # _get_llm_token_counts) so it can never blank out the attributes
        # yielded above: without this, the exception would reach
        # _finish_tracing, which drops every attribute on the span.
        try:
            yield from _get_token_counts(snapshot.usage)
        except Exception:
            logger.exception("Failed to get token counts from streaming snapshot.")
