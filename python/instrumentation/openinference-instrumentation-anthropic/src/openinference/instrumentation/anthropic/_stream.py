from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt.proxies import ObjectProxy

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.anthropic._utils import (
    _as_output_attributes,
    _finish_tracing,
    _ValueAndType,
)
from openinference.instrumentation.anthropic._with_span import _WithSpan
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from anthropic import Stream
    from anthropic.types import Completion, RawMessageStreamEvent


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


class _Stream(ObjectProxy):  # type: ignore[misc,name-defined,type-arg,unused-ignore]
    __slots__ = (
        "_response_accumulator",
        "_with_span",
    )

    def __init__(
        self,
        stream: "Stream[Completion]",
        with_span: _WithSpan,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _ResponseAccumulator()
        self._with_span = with_span

    def __iter__(self) -> Iterator["Completion"]:
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

    async def __aiter__(self) -> AsyncIterator["Completion"]:
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
            has_attributes=_ResponseExtractor(response_accumulator=self._response_accumulator),
            status=status,
        )


class _ResponseAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
    )

    def __init__(self) -> None:
        self._is_null = True
        self._values = _ValuesAccumulator(
            completion=_StringAccumulator(),
            stop=_SimpleStringReplace(),
            stop_reason=_SimpleStringReplace(),
        )

    def process_chunk(self, chunk: "Completion") -> None:
        self._is_null = False
        values = chunk.model_dump(exclude_unset=True, warnings=False)
        self._values += values

    def _result(self) -> Optional[Dict[str, Any]]:
        if self._is_null:
            return None
        return dict(self._values)


class _ResponseExtractor:
    __slots__ = ("_response_accumulator",)

    def __init__(
        self,
        response_accumulator: _ResponseAccumulator,
    ) -> None:
        self._response_accumulator = response_accumulator

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._response_accumulator._result()):
            return
        json_string = safe_json_dumps(result)
        yield from _as_output_attributes(
            _ValueAndType(json_string, OpenInferenceMimeTypeValues.JSON)
        )
        if completion := result.get("completion", ""):
            yield SpanAttributes.LLM_OUTPUT_MESSAGES, completion


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
        tool_idx = 0
        for block in snapshot.content:
            if block.type == "text":
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}",
                    block.text,
                )
            elif block.type == "tool_use":
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                    block.name,
                )
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                    safe_json_dumps(block.input),
                )
                tool_idx += 1
        usage = snapshot.usage
        prompt_tokens = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )
        if prompt_tokens:
            yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
        if usage.output_tokens:
            yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage.output_tokens
        if total := prompt_tokens + (usage.output_tokens or 0):
            yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total


class _ValuesAccumulator:
    __slots__ = ("_values",)

    def __init__(self, **values: Any) -> None:
        self._values: Dict[str, Any] = values

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for key, value in self._values.items():
            if value is None:
                continue
            if isinstance(value, _ValuesAccumulator):
                if dict_value := dict(value):
                    yield key, dict_value
            elif isinstance(value, _SimpleStringReplace):
                if str_value := str(value):
                    yield key, str_value
            elif isinstance(value, _StringAccumulator):
                if str_value := str(value):
                    yield key, str_value
            else:
                yield key, value

    def __iadd__(self, values: Optional[Mapping[str, Any]]) -> "_ValuesAccumulator":
        if not values:
            return self
        for key in self._values.keys():
            if (value := values.get(key)) is None:
                continue
            self_value = self._values[key]
            if isinstance(self_value, _ValuesAccumulator):
                if isinstance(value, Mapping):
                    self_value += value
            elif isinstance(self_value, _StringAccumulator):
                if isinstance(value, str):
                    self_value += value
            elif isinstance(self_value, _SimpleStringReplace):
                if isinstance(value, str):
                    self_value += value
            elif isinstance(self_value, List) and isinstance(value, Iterable):
                self_value.extend(value)
            else:
                self._values[key] = value  # replacement
        for key in values.keys():
            if key in self._values or (value := values[key]) is None:
                continue
            value = deepcopy(value)
            if isinstance(value, Mapping):
                value = _ValuesAccumulator(**value)
            self._values[key] = value  # new entry
        return self


class _StringAccumulator:
    __slots__ = ("_fragments",)

    def __init__(self) -> None:
        self._fragments: List[str] = []

    def __str__(self) -> str:
        return "".join(self._fragments)

    def __iadd__(self, value: Optional[str]) -> "_StringAccumulator":
        if not value:
            return self
        self._fragments.append(value)
        return self


class _SimpleStringReplace:
    __slots__ = ("_str_val",)

    def __init__(self) -> None:
        self._str_val: str = ""

    def __str__(self) -> str:
        return self._str_val

    def __iadd__(self, value: Optional[str]) -> "_SimpleStringReplace":
        if not value:
            return self
        self._str_val = value
        return self
