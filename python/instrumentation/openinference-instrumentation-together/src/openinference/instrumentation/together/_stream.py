import logging
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.together._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.together._utils import _finish_tracing
from openinference.instrumentation.together._with_span import _WithSpan
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ChunkAccumulator:
    """Accumulates streamed chat-completion chunks into span attributes."""

    def __init__(self, response_extractor: _ResponseAttributesExtractor) -> None:
        self._response_extractor = response_extractor
        self._model: Any = None
        self._usage: Any = None
        self._roles: Dict[int, str] = {}
        self._contents: Dict[int, List[str]] = defaultdict(list)
        self._tool_calls: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        self._completion: Optional[SimpleNamespace] = None

    def process_chunk(self, chunk: Any) -> None:
        try:
            if model := getattr(chunk, "model", None):
                self._model = model
            if usage := getattr(chunk, "usage", None):
                self._usage = usage
            for choice in getattr(chunk, "choices", None) or ():
                index = getattr(choice, "index", None) or 0
                if (delta := getattr(choice, "delta", None)) is None:
                    continue
                if (role := getattr(delta, "role", None)) and index not in self._roles:
                    self._roles[index] = role
                if content := getattr(delta, "content", None):
                    self._contents[index].append(content)
                for position, tool_call in enumerate(getattr(delta, "tool_calls", None) or ()):
                    tool_index = getattr(tool_call, "index", None)
                    if tool_index is None:
                        tool_index = position
                    entry = self._tool_calls[index].setdefault(
                        tool_index, {"id": None, "name": None, "arguments": []}
                    )
                    if tool_call_id := getattr(tool_call, "id", None):
                        entry["id"] = tool_call_id
                    if function := getattr(tool_call, "function", None):
                        if name := getattr(function, "name", None):
                            entry["name"] = name
                        if arguments := getattr(function, "arguments", None):
                            entry["arguments"].append(arguments)
        except Exception:
            logger.exception(f"Failed to process stream chunk of type {type(chunk)}")

    def _get_completion(self) -> SimpleNamespace:
        """Materializes the accumulated chunks into a chat-completion-shaped object."""
        if self._completion is not None:
            return self._completion
        choices = []
        for index in sorted(self._roles.keys() | self._contents.keys() | self._tool_calls.keys()):
            tool_calls = [
                SimpleNamespace(
                    id=entry["id"],
                    function=SimpleNamespace(
                        name=entry["name"], arguments="".join(entry["arguments"])
                    ),
                )
                for _, entry in sorted(self._tool_calls.get(index, {}).items())
            ]
            message = SimpleNamespace(
                role=self._roles.get(index),
                content="".join(self._contents.get(index, [])) or None,
                function_call=None,
                tool_calls=tool_calls or None,
            )
            choices.append(SimpleNamespace(index=index, message=message))
        self._completion = SimpleNamespace(model=self._model, usage=self._usage, choices=choices)
        return self._completion

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        choices = self._get_completion().choices
        if len(choices) == 1 and choices[0].message.content and not choices[0].message.tool_calls:
            yield SpanAttributes.OUTPUT_VALUE, choices[0].message.content
        else:
            messages = [
                {
                    key: value
                    for key, value in {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": tool_call.function.arguments,
                                },
                            }
                            for tool_call in choice.message.tool_calls or ()
                        ]
                        or None,
                    }.items()
                    if value is not None
                }
                for choice in choices
            ]
            yield SpanAttributes.OUTPUT_VALUE, safe_json_dumps(messages)
            yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        yield from self._response_extractor.get_extra_attributes(response=self._get_completion())


class _Stream(ObjectProxy):  # type: ignore[misc,type-arg,unused-ignore]
    """Wraps ``together.Stream`` and ``together.AsyncStream`` to finish the span
    when the stream is consumed (or the stream's context manager exits)."""

    __slots__ = ("_self_with_span", "_self_accumulator")

    def __init__(
        self,
        stream: Any,
        with_span: _WithSpan,
        response_extractor: _ResponseAttributesExtractor,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_accumulator = _ChunkAccumulator(response_extractor)

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = self.__wrapped__.__next__()
        except StopIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_error(exception)
            raise
        self._self_accumulator.process_chunk(chunk)
        return chunk

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_error(exception)
            raise
        self._self_accumulator.process_chunk(chunk)
        return chunk

    def __enter__(self) -> "_Stream":
        self.__wrapped__.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        result = self.__wrapped__.__exit__(*args)
        self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
        return result

    async def __aenter__(self) -> "_Stream":
        await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        result = await self.__wrapped__.__aexit__(*args)
        self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
        return result

    def _finish_error(self, exception: Exception) -> None:
        self._self_with_span.record_exception(exception)
        self._finish(
            trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def _finish(self, status: trace_api.Status) -> None:
        if self._self_with_span.is_finished:
            return
        _finish_tracing(
            status=status,
            with_span=self._self_with_span,
            attributes=self._self_accumulator.get_attributes(),
            extra_attributes=self._self_accumulator.get_extra_attributes(),
        )
