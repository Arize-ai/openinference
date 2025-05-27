import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
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
    _ValueAndType,
)
from openinference.semconv.trace import OpenInferenceMimeTypeValues

if TYPE_CHECKING:
    from openai.types import Completion
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

__all__ = (
    "_CompletionAccumulator",
    "_ChatCompletionAccumulator",
)


class _CanGetAttributesFromResponse(Protocol):
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]: ...


class _ChatCompletionAccumulator:
    __slots__ = (
        "_is_null",
        "_values",
        "_cached_result",
        "_request_parameters",
        "_response_attributes_extractor",
        "_chat_completion_type",
        "_has_usage",  # Track if we've seen usage data
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
        self._cached_result = None
        self._has_usage = False
        self._values = _ValuesAccumulator(
            choices=_IndexedAccumulator(
                lambda: _ValuesAccumulator(
                    message=_ValuesAccumulator(
                        role=_StringAccumulator(),
                        content=_StringAccumulator(),
                        function_call=_ValuesAccumulator(
                            name=_StringAccumulator(),
                            arguments=_StringAccumulator(),
                        ),
                        tool_calls=_IndexedAccumulator(
                            lambda: _ValuesAccumulator(
                                id=_StringAccumulator(),
                                type=_StringAccumulator(),
                                function=_ValuesAccumulator(
                                    name=_StringAccumulator(),
                                    arguments=_StringAccumulator(),
                                ),
                            ),
                        ),
                    ),
                    finish_reason=_StringAccumulator(),
                ),
            ),
            usage=_ValuesAccumulator(
                prompt_tokens=_IntAccumulator(),
                completion_tokens=_IntAccumulator(),
                total_tokens=_IntAccumulator(),
                prompt_tokens_details=_ValuesAccumulator(
                    cached_tokens=_IntAccumulator(),
                    cache_input=_IntAccumulator(),
                    audio_tokens=_IntAccumulator(),
                ),
                completion_tokens_details=_ValuesAccumulator(
                    reasoning_tokens=_IntAccumulator(),
                    audio_tokens=_IntAccumulator(),
                    accepted_prediction_tokens=_IntAccumulator(),
                    rejected_prediction_tokens=_IntAccumulator(),
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

        # Check if this chunk has usage data
        if "usage" in values:
            self._has_usage = True

        # For streaming responses, we need to handle tool calls specially
        if "choices" in values:
            for choice in values["choices"]:
                if "delta" in choice:
                    delta = choice["delta"]
                    if "tool_calls" in delta:
                        # For tool calls, we need to merge them with existing tool calls
                        if "choices" not in self._values:
                            self._values["choices"] = _IndexedAccumulator(
                                lambda: _ValuesAccumulator(
                                    message=_ValuesAccumulator(
                                        tool_calls=_IndexedAccumulator(
                                            lambda: _ValuesAccumulator(
                                                id=_StringAccumulator(),
                                                type=_StringAccumulator(),
                                                function=_ValuesAccumulator(
                                                    name=_StringAccumulator(),
                                                    arguments=_StringAccumulator(),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            )
                        # Merge tool calls
                        for tool_call in delta["tool_calls"]:
                            idx = tool_call.get("index", 0)
                            if "choices" in self._values and idx < len(self._values["choices"]):
                                if "message" not in self._values["choices"][idx]:
                                    self._values["choices"][idx]["message"] = _ValuesAccumulator(
                                        tool_calls=_IndexedAccumulator(
                                            lambda: _ValuesAccumulator(
                                                id=_StringAccumulator(),
                                                type=_StringAccumulator(),
                                                function=_ValuesAccumulator(
                                                    name=_StringAccumulator(),
                                                    arguments=_StringAccumulator(),
                                                ),
                                            ),
                                        ),
                                    )
                                if "tool_calls" not in self._values["choices"][idx]["message"]:
                                    self._values["choices"][idx]["message"]["tool_calls"] = (
                                        _IndexedAccumulator(
                                            lambda: _ValuesAccumulator(
                                                id=_StringAccumulator(),
                                                type=_StringAccumulator(),
                                                function=_ValuesAccumulator(
                                                    name=_StringAccumulator(),
                                                    arguments=_StringAccumulator(),
                                                ),
                                            ),
                                        )
                                    )
                                # Update the tool call
                                if "id" in tool_call:
                                    self._values["choices"][idx]["message"]["tool_calls"][idx][
                                        "id"
                                    ] = tool_call["id"]
                                if "type" in tool_call:
                                    self._values["choices"][idx]["message"]["tool_calls"][idx][
                                        "type"
                                    ] = tool_call["type"]
                                if "function" in tool_call:
                                    if "name" in tool_call["function"]:
                                        self._values["choices"][idx]["message"]["tool_calls"][idx][
                                            "function"
                                        ]["name"] = tool_call["function"]["name"]
                                    if "arguments" in tool_call["function"]:
                                        self._values["choices"][idx]["message"]["tool_calls"][idx][
                                            "function"
                                        ]["arguments"] = tool_call["function"]["arguments"]
                    else:
                        # For non-tool-call deltas, just update normally
                        self._values += values

    def _result(self) -> Optional[Dict[str, Any]]:
        if self._is_null:
            return None
        if not self._cached_result:
            self._cached_result = dict(self._values)
            # If we haven't seen usage data in streaming mode, don't include it
            if not self._has_usage and "usage" in self._cached_result:
                del self._cached_result["usage"]
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


class _StringAccumulator:
    def __init__(self) -> None:
        self._value: Optional[str] = None

    def __iadd__(self, value: Optional[str]) -> "_StringAccumulator":
        if value is not None:
            self._value = value
        return self

    def __bool__(self) -> bool:
        return self._value is not None

    def __str__(self) -> str:
        return str(self._value) if self._value is not None else ""

    def __repr__(self) -> str:
        return repr(self._value)


class _IntAccumulator:
    def __init__(self) -> None:
        self._value: Optional[int] = None

    def __iadd__(self, value: Optional[int]) -> "_IntAccumulator":
        if value is not None:
            self._value = value
        return self

    def __bool__(self) -> bool:
        return self._value is not None

    def __int__(self) -> int:
        return self._value if self._value is not None else 0

    def __repr__(self) -> str:
        return repr(self._value)


class _ValuesAccumulator(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __iadd__(self, values: Mapping[str, Any]) -> "_ValuesAccumulator":
        for key, value in values.items():
            if key not in self:
                self[key] = value
            elif isinstance(self[key], _ValuesAccumulator):
                self[key] += value
            elif isinstance(self[key], _StringAccumulator):
                self[key] += value
            elif isinstance(self[key], _IntAccumulator):
                self[key] += value
            elif isinstance(self[key], _IndexedAccumulator):
                self[key] += value
        return self


class _IndexedAccumulator(list):
    def __init__(self, factory: Callable[[], Any]) -> None:
        super().__init__()
        self._factory = factory

    def __iadd__(self, values: Iterable[Any]) -> "_IndexedAccumulator":
        for value in values:
            if isinstance(value, Mapping):
                if "index" in value:
                    idx = value["index"]
                    while len(self) <= idx:
                        self.append(self._factory())
                    self[idx] += value
            else:
                self.append(value)
        return self
