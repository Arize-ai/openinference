from collections import defaultdict
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

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
    from anthropic.types.raw_content_block_delta_event import RawContentBlockDeltaEvent
    from anthropic.types.text_block import TextBlock
    from anthropic.types.tool_use_block import ToolUseBlock


class _Stream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
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

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._response_accumulator._result()):
            return
        if completion := result.get("completion", ""):
            yield SpanAttributes.LLM_OUTPUT_MESSAGES, completion


class _MessagesStream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
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
            has_attributes=_MessageResponseExtractor(
                response_accumulator=self._response_accumulator
            ),
            status=status,
        )


class _MessageResponseAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_current_message_idx",
        "_current_content_block_type",
    )

    def __init__(self) -> None:
        self._is_null = True
        self._current_message_idx = -1
        self._current_content_block_type: Union["TextBlock", "ToolUseBlock", None] = None
        self._values = _ValuesAccumulator(
            messages=_IndexedAccumulator(
                lambda: _ValuesAccumulator(
                    role=_SimpleStringReplace(),
                    content=_IndexedAccumulator(
                        lambda: _ValuesAccumulator(
                            type=_SimpleStringReplace(),
                            text=_StringAccumulator(),
                            tool_name=_SimpleStringReplace(),
                            tool_input=_StringAccumulator(),
                        ),
                    ),
                    stop_reason=_SimpleStringReplace(),
                    input_tokens=_SimpleStringReplace(),
                    output_tokens=_SimpleStringReplace(),
                ),
            ),
        )

    def process_chunk(self, chunk: "RawContentBlockDeltaEvent") -> None:
        from anthropic.types.raw_content_block_delta_event import RawContentBlockDeltaEvent
        from anthropic.types.raw_content_block_start_event import RawContentBlockStartEvent
        from anthropic.types.raw_message_delta_event import RawMessageDeltaEvent
        from anthropic.types.raw_message_start_event import RawMessageStartEvent
        from anthropic.types.text_block import TextBlock
        from anthropic.types.tool_use_block import ToolUseBlock

        self._is_null = False
        if isinstance(chunk, RawMessageStartEvent):
            self._current_message_idx += 1
            value = {
                "messages": {
                    "index": str(self._current_message_idx),
                    "role": chunk.message.role,
                    "input_tokens": str(chunk.message.usage.input_tokens),
                }
            }
            self._values += value
        elif isinstance(chunk, RawContentBlockStartEvent):
            self._current_content_block_type = chunk.content_block
        elif isinstance(chunk, RawContentBlockDeltaEvent):
            if isinstance(self._current_content_block_type, TextBlock):
                value = {
                    "messages": {
                        "index": str(self._current_message_idx),
                        "content": {
                            "index": chunk.index,
                            "type": self._current_content_block_type.type,
                            "text": chunk.delta.text,  # type: ignore
                        },
                    }
                }
                self._values += value
            elif isinstance(self._current_content_block_type, ToolUseBlock):
                value = {
                    "messages": {
                        "index": str(self._current_message_idx),
                        "content": {
                            "index": chunk.index,
                            "type": self._current_content_block_type.type,
                            "tool_name": self._current_content_block_type.name,
                            "tool_input": chunk.delta.partial_json,  # type: ignore
                        },
                    }
                }
                self._values += value
        elif isinstance(chunk, RawMessageDeltaEvent):
            value = {
                "messages": {
                    "index": str(self._current_message_idx),
                    "stop_reason": chunk.delta.stop_reason,
                    "output_tokens": str(chunk.usage.output_tokens),
                }
            }
            self._values += value

    def _result(self) -> Optional[Dict[str, Any]]:
        if self._is_null:
            return None
        return dict(self._values)


class _MessageResponseExtractor:
    __slots__ = ("_response_accumulator",)

    def __init__(
        self,
        response_accumulator: _MessageResponseAccumulator,
    ) -> None:
        self._response_accumulator = response_accumulator

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._response_accumulator._result()):
            return
        json_string = safe_json_dumps(result)
        yield from _as_output_attributes(
            _ValueAndType(json_string, OpenInferenceMimeTypeValues.JSON)
        )

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._response_accumulator._result()):
            return
        messages = result.get("messages", [])
        idx = 0
        total_completion_token_count = 0
        total_prompt_token_count = 0
        # TODO(harrison): figure out if we should always assume messages is 1.
        # The current non streaming implementation assumes the same
        for message in messages:
            if role := message.get("role"):
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_ROLE}",
                    role,
                )
            if output_tokens := message.get("output_tokens"):
                total_completion_token_count += int(output_tokens)
            if input_tokens := message.get("input_tokens"):
                total_prompt_token_count += int(input_tokens)

            # TODO(harrison): figure out if we should always assume the first message
            #  will always be a message output generally this block feels really
            #  brittle to imitate the current non streaming implementation.
            tool_idx = 0
            for content in message.get("content", []):
                # this is the current assumption of the non streaming implementation.
                if (content_type := content.get("type")) == "text":
                    yield (
                        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_CONTENT}",
                        content.get("text", ""),
                    )
                elif content_type == "tool_use":
                    yield (
                        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                        content.get("tool_name", ""),
                    )
                    yield (
                        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_idx}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        content.get("tool_input", "{}"),
                    )
                    tool_idx += 1
            idx += 1
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, total_completion_token_count
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, total_prompt_token_count
        yield (
            SpanAttributes.LLM_TOKEN_COUNT_TOTAL,
            total_completion_token_count + total_prompt_token_count,
        )


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
            elif isinstance(self_value, _IndexedAccumulator):
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


class _IndexedAccumulator:
    __slots__ = ("_indexed",)

    def __init__(self, factory: Callable[[], _ValuesAccumulator]) -> None:
        self._indexed: DefaultDict[int, _ValuesAccumulator] = defaultdict(factory)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _, values in sorted(self._indexed.items()):
            yield dict(values)

    def __iadd__(self, values: Optional[Mapping[str, Any]]) -> "_IndexedAccumulator":
        if not values or not hasattr(values, "get") or (index := values.get("index")) is None:
            return self
        self._indexed[index] += values
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
