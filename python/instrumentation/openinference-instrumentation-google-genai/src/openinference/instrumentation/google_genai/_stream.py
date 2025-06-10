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
from openinference.instrumentation.google_genai._utils import (
    _as_output_attributes,
    _finish_tracing,
    _ValueAndType,
)
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse


class _Stream(ObjectProxy):  # type: ignore
    __slots__ = (
        "_response_accumulator",
        "_with_span",
        "_is_finished",
    )

    def __init__(
        self,
        stream: Iterator["GenerateContentResponse"],
        with_span: _WithSpan,
    ) -> None:
        super().__init__(stream)
        self._response_accumulator = _ResponseAccumulator()
        self._with_span = with_span

    def __iter__(self) -> Iterator["GenerateContentResponse"]:
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

    async def __aiter__(self) -> AsyncIterator["GenerateContentResponse"]:
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
        response_extractor = _ResponseExtractor(response_accumulator=self._response_accumulator)
        _finish_tracing(
            with_span=self._with_span,
            attributes=response_extractor.get_attributes(),
            extra_attributes=response_extractor.get_extra_attributes(),
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
            candidates=_IndexedAccumulator(
                lambda: _ValuesAccumulator(
                    content=_ValuesAccumulator(
                        parts=_IndexedAccumulator(
                            lambda: _ValuesAccumulator(
                                text=_StringAccumulator(),
                            ),
                        ),
                        role=_SimpleStringReplace(),
                    ),
                    finish_reason=_SimpleStringReplace(),
                ),
            ),
            usage_metadata=_DictReplace(),
            model_version=_SimpleStringReplace(),
        )

    def process_chunk(self, chunk: "GenerateContentResponse") -> None:
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
        if model_version := result.get("model_version"):
            yield SpanAttributes.LLM_MODEL_NAME, model_version
        if usage_metadata := result.get("usage_metadata"):
            if prompt_token_count := usage_metadata.get("prompt_token_count"):
                yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(prompt_token_count)
            if candidates_token_count := usage_metadata.get("candidates_token_count"):
                yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(candidates_token_count)
            if total_token_count := usage_metadata.get("total_token_count"):
                yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(total_token_count)
        if candidates := result.get("candidates"):
            for idx, candidate in enumerate(candidates):
                if content := candidate.get("content"):
                    if role := content.get("role"):
                        yield (
                            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_ROLE}",
                            role,
                        )
                    if parts := content.get("parts"):
                        text_parts = []
                        for part in parts:
                            if text := part.get("text"):
                                text_parts.append(text)
                        if text_parts:
                            yield (
                                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{idx}.{MessageAttributes.MESSAGE_CONTENT}",
                                "".join(text_parts),
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
            elif isinstance(value, _IndexedAccumulator):
                yield key, list(value)
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
        # If the value is a list of strings, extend with all of them
        if isinstance(value, list):
            self._fragments.extend(value)
        # If the value is a dict with a text field, add just the text
        elif isinstance(value, dict) and "text" in value:
            self._fragments.append(value["text"])
        # Otherwise treat as a string
        else:
            self._fragments.append(str(value))
        return self


class _IndexedAccumulator:
    __slots__ = ("_indexed",)

    def __init__(self, factory: Callable[[], _ValuesAccumulator]) -> None:
        self._indexed: DefaultDict[int, _ValuesAccumulator] = defaultdict(factory)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for _, values in sorted(self._indexed.items()):
            yield dict(values)

    def __iadd__(
        self, values: Optional[Union[Mapping[str, Any], List[Any]]]
    ) -> "_IndexedAccumulator":
        if not values:
            return self
        if isinstance(values, Mapping):
            values = [values]
        for v in values:
            if v and hasattr(v, "get"):
                self._indexed[v.get("index") or 0] += v
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


class _DictReplace:
    __slots__ = ("_val",)

    def __init__(self) -> None:
        self._val: Mapping[Any, Any] = {}

    def __iadd__(self, value: Optional[Mapping[Any, Any]]) -> "_DictReplace":
        if not value:
            return self
        self._val = value
        return self
