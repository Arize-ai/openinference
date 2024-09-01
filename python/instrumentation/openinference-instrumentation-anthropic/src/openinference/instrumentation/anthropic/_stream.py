from copy import deepcopy
from typing import (
    Any,
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
from wrapt import ObjectProxy

from anthropic import Stream
from anthropic.types import (
    Completion,
    RawMessageStreamEvent,
)
from anthropic.types.raw_content_block_delta_event import RawContentBlockDeltaEvent
from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.anthropic._utils import (
    _as_output_attributes,
    _finish_tracing,
    _ValueAndType,
)
from openinference.instrumentation.anthropic._with_span import _WithSpan
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)


class _Stream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
    )

    def __init__(
        self,
        stream: Stream[Completion],
        with_span: _WithSpan,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _ResponseAccumulator()
        self._with_span = with_span

    def __iter__(self) -> Iterator[Completion]:
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

    def process_chunk(self, chunk: Completion) -> None:
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
            stream: Stream[RawMessageStreamEvent],
            with_span: _WithSpan,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _MessageResponseAccumulator()
        self._with_span = with_span

    def __iter__(self) -> Iterator[RawMessageStreamEvent]:
        try:
            for item in self.__wrapped__:
                if isinstance(item, RawContentBlockDeltaEvent):
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
            has_attributes=_MessageResponseExtractor(response_accumulator=self._response_accumulator),
            status=status,
        )


class _MessageResponseAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
    )

    def __init__(self) -> None:
        self._is_null = True
        self._values = _ValuesAccumulator(
            delta=_ValuesAccumulator(
                text=_StringAccumulator(),
            ),
        )

    def process_chunk(self, chunk: RawContentBlockDeltaEvent) -> None:
        self._is_null = False
        values = chunk.model_dump(exclude_unset=True, warnings=False)
        self._values += values

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
        if completion := result.get("completion", ""):
            yield SpanAttributes.LLM_OUTPUT_MESSAGES, completion

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

    def __str__(self) -> str:
        return self._str_val

    def __iadd__(self, value: Optional[str]) -> "_SimpleStringReplace":
        if not value:
            return self
        self._str_val = value
        return self
