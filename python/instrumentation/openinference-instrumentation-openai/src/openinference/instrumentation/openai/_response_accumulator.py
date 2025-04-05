import warnings
from collections import defaultdict
from copy import deepcopy
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    Type,
)

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.openai._utils import (
    _as_output_attributes,
    _io_value_and_type,
    _ValueAndType,
)
from openinference.semconv.trace import OpenInferenceMimeTypeValues

if TYPE_CHECKING:
    from openai.types import Completion
    from openai.types.chat import ChatCompletion, ChatCompletionChunk
    from openai.types.responses.response import Response
    from openai.types.responses.response_completed_event import ResponseCompletedEvent

__all__ = (
    "_CompletionAccumulator",
    "_ChatCompletionAccumulator",
    "_ResponsesAccumulator",
)


class _CanGetAttributesFromResponse(Protocol):
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]: ...


class _ResponsesAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
        "_request_parameters",
        "_response_attributes_extractor",
        "_chat_completion_type",
    )

    def __init__(
        self,
        request_parameters: Mapping[str, Any],
        chat_completion_type: Type["Response"],
        response_attributes_extractor: Optional[_CanGetAttributesFromResponse] = None,
    ) -> None:
        self._chat_completion_type = chat_completion_type
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor
        self._is_null = True
        self._cached_result: Optional[Dict[str, Any]] = None
        self._values: Optional["ResponseCompletedEvent"] = None

    def process_chunk(self, chunk: Any) -> None:
        if type(chunk).__name__ == "ResponseCompletedEvent":
            self._is_null = False
            self._cached_result = None
            self._values = chunk

    def _result(self) -> Any:
        if self._is_null:
            return None
        if self._values:
            return self._values.response

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        yield from _as_output_attributes(
            _io_value_and_type(result),
        )

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        if self._response_attributes_extractor:
            yield from self._response_attributes_extractor.get_attributes_from_response(
                response=result,
                request_parameters=self._request_parameters,
            )


class _ChatCompletionAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
        "_request_parameters",
        "_response_attributes_extractor",
        "_chat_completion_type",
    )

    def __init__(
        self,
        request_parameters: Mapping[str, Any],
        chat_completion_type: Type["ChatCompletion"],
        response_attributes_extractor: Optional[_CanGetAttributesFromResponse] = None,
    ) -> None:
        self._chat_completion_type = chat_completion_type
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor
        self._is_null = True
        self._cached_result: Optional[Dict[str, Any]] = None
        self._values = _ValuesAccumulator(
            choices=_IndexedAccumulator(
                lambda: _ValuesAccumulator(
                    message=_ValuesAccumulator(
                        content=_StringAccumulator(),
                        function_call=_ValuesAccumulator(arguments=_StringAccumulator()),
                        tool_calls=_IndexedAccumulator(
                            lambda: _ValuesAccumulator(
                                function=_ValuesAccumulator(arguments=_StringAccumulator()),
                            )
                        ),
                    ),
                ),
            ),
        )

    def process_chunk(self, chunk: "ChatCompletionChunk") -> None:
        self._is_null = False
        self._cached_result = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # `warnings=False` in `model_dump()` is only supported in Pydantic v2
            values = chunk.model_dump(exclude_unset=True)
        for choice in values.get("choices", ()):
            if delta := choice.pop("delta", None):
                choice["message"] = delta
        self._values += values

    def _result(self) -> Optional[Dict[str, Any]]:
        if self._is_null:
            return None
        if not self._cached_result:
            self._cached_result = dict(self._values)
        return self._cached_result

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        json_string = safe_json_dumps(result)
        yield from _as_output_attributes(
            _ValueAndType(json_string, OpenInferenceMimeTypeValues.JSON),
        )

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        if self._response_attributes_extractor:
            yield from self._response_attributes_extractor.get_attributes_from_response(
                self._chat_completion_type.construct(**result),
                self._request_parameters,
            )


class _CompletionAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
        "_request_parameters",
        "_response_attributes_extractor",
        "_completion_type",
    )

    def __init__(
        self,
        request_parameters: Mapping[str, Any],
        completion_type: Type["Completion"],
        response_attributes_extractor: Optional[_CanGetAttributesFromResponse] = None,
    ) -> None:
        self._completion_type = completion_type
        self._request_parameters = request_parameters
        self._response_attributes_extractor = response_attributes_extractor
        self._is_null = True
        self._cached_result: Optional[Dict[str, Any]] = None
        self._values = _ValuesAccumulator(
            choices=_IndexedAccumulator(lambda: _ValuesAccumulator(text=_StringAccumulator())),
        )

    def process_chunk(self, chunk: "Completion") -> None:
        self._is_null = False
        self._cached_result = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # `warnings=False` in `model_dump()` is only supported in Pydantic v2
            values = chunk.model_dump(exclude_unset=True)
        self._values += values

    def _result(self) -> Optional[Dict[str, Any]]:
        if self._is_null:
            return None
        if not self._cached_result:
            self._cached_result = dict(self._values)
        return self._cached_result

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        json_string = safe_json_dumps(result)
        yield from _as_output_attributes(
            _ValueAndType(json_string, OpenInferenceMimeTypeValues.JSON),
        )

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        if self._response_attributes_extractor:
            yield from self._response_attributes_extractor.get_attributes_from_response(
                self._completion_type.construct(**result),
                self._request_parameters,
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
            elif isinstance(value, _IndexedAccumulator):
                if list_value := list(value):
                    yield key, list_value
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
            elif isinstance(self_value, _IndexedAccumulator):
                if isinstance(value, Iterable):
                    for v in value:
                        self_value += v
                else:
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
