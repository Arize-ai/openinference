import json
import warnings
from collections import defaultdict
from copy import deepcopy
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

from openinference.instrumentation.openai._extra_attributes_from_response import (
    _get_extra_attributes_from_response,
)
from openinference.instrumentation.openai._utils import (
    _as_output_attributes,
    _MimeType,
    _ValueAndType,
)
from opentelemetry.util.types import AttributeValue
from typing_extensions import TypeAlias

from openai.types import Completion
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)

__all__ = (
    "_CompletionAccumulator",
    "_ChatCompletionAccumulator",
)

_ChoiceIndex: TypeAlias = int


class _ChatCompletionAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
    )

    def __init__(self) -> None:
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

    def process_chunk(self, chunk: ChatCompletionChunk) -> None:
        if not isinstance(chunk, ChatCompletionChunk):
            return
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
        json_string = json.dumps(result)
        yield from _as_output_attributes(_ValueAndType(json_string, _MimeType.application_json))

    def get_extra_attributes(
        self,
        request_options: Mapping[str, Any] = MappingProxyType({}),
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        yield from _get_extra_attributes_from_response(
            ChatCompletion.construct(**result),
            request_options,
        )


class _CompletionAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
    )

    def __init__(self) -> None:
        self._is_null = True
        self._cached_result: Optional[Dict[str, Any]] = None
        self._values = _ValuesAccumulator(
            choices=_IndexedAccumulator(lambda: _ValuesAccumulator(text=_StringAccumulator())),
        )

    def process_chunk(self, chunk: Completion) -> None:
        if not isinstance(chunk, Completion):
            return
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
        json_string = json.dumps(result)
        yield from _as_output_attributes(_ValueAndType(json_string, _MimeType.application_json))

    def get_extra_attributes(
        self,
        request_options: Mapping[str, Any] = MappingProxyType({}),
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not (result := self._result()):
            return
        yield from _get_extra_attributes_from_response(
            Completion.construct(**result),
            request_options,
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
