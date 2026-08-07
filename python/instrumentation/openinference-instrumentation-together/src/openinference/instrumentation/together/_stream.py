import logging
from collections import defaultdict
from typing import Any, AsyncIterator, Dict, Iterator, List, Tuple

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.together._utils import _finish_tracing
from openinference.instrumentation.together._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ChunkAccumulator:
    """Accumulates streamed chat-completion chunks into span attributes."""

    def __init__(self) -> None:
        self._model: Any = None
        self._usage: Any = None
        self._roles: Dict[int, str] = {}
        self._contents: Dict[int, List[str]] = defaultdict(list)
        self._tool_calls: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)

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

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        content = "".join(self._contents.get(0, []))
        if content and not self._tool_calls:
            yield SpanAttributes.OUTPUT_VALUE, content
        else:
            yield SpanAttributes.OUTPUT_VALUE, safe_json_dumps(self._as_messages())
            yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if self._model:
            yield SpanAttributes.LLM_MODEL_NAME, self._model
        if usage := self._usage:
            if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
                yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens
            if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
                yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
            if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
                yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
        for index in sorted(self._roles.keys() | self._contents.keys() | self._tool_calls.keys()):
            prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}"
            if role := self._roles.get(index):
                yield f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", role
            if content := "".join(self._contents.get(index, [])):
                yield f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content
            for tool_index, entry in sorted(self._tool_calls.get(index, {}).items()):
                tool_prefix = f"{prefix}.{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_index}"
                if entry["id"] is not None:
                    yield f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_ID}", entry["id"]
                if entry["name"] is not None:
                    yield (
                        f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                        entry["name"],
                    )
                if arguments := "".join(entry["arguments"]):
                    yield (
                        f"{tool_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
                    )

    def _as_messages(self) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        for index in sorted(self._roles.keys() | self._contents.keys() | self._tool_calls.keys()):
            message: Dict[str, Any] = {}
            if role := self._roles.get(index):
                message["role"] = role
            if content := "".join(self._contents.get(index, [])):
                message["content"] = content
            if tool_calls := self._tool_calls.get(index):
                message["tool_calls"] = [
                    {
                        "id": entry["id"],
                        "function": {
                            "name": entry["name"],
                            "arguments": "".join(entry["arguments"]),
                        },
                    }
                    for _, entry in sorted(tool_calls.items())
                ]
            messages.append(message)
        return messages


class _StreamStateMixin:
    """Span bookkeeping shared by the sync and async stream proxies."""

    _self_with_span: _WithSpan
    _self_accumulator: _ChunkAccumulator

    def _process_chunk(self, chunk: Any) -> None:
        self._self_accumulator.process_chunk(chunk)

    def _finish_success(self) -> None:
        if self._self_with_span.is_finished:
            return
        _finish_tracing(
            status=trace_api.Status(status_code=trace_api.StatusCode.OK),
            with_span=self._self_with_span,
            attributes=self._self_accumulator.get_attributes(),
            extra_attributes=self._self_accumulator.get_extra_attributes(),
        )

    def _finish_error(self, exception: Exception) -> None:
        if self._self_with_span.is_finished:
            return
        self._self_with_span.record_exception(exception)
        _finish_tracing(
            status=trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            ),
            with_span=self._self_with_span,
            attributes=self._self_accumulator.get_attributes(),
            extra_attributes=self._self_accumulator.get_extra_attributes(),
        )


class _Stream(ObjectProxy, _StreamStateMixin):  # type: ignore[misc,type-arg,unused-ignore]
    """Wraps ``together.Stream`` to finish the span when the stream is consumed."""

    def __init__(self, stream: Any, with_span: _WithSpan) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_accumulator = _ChunkAccumulator()

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk = self.__wrapped__.__next__()
        except StopIteration:
            self._finish_success()
            raise
        except Exception as exception:
            self._finish_error(exception)
            raise
        self._process_chunk(chunk)
        return chunk

    def __enter__(self) -> "_Stream":
        self.__wrapped__.__enter__()
        return self

    def __exit__(self, *args: Any) -> Any:
        result = self.__wrapped__.__exit__(*args)
        self._finish_success()
        return result

    def close(self) -> None:
        self.__wrapped__.close()
        self._finish_success()


class _AsyncStream(ObjectProxy, _StreamStateMixin):  # type: ignore[misc,type-arg,unused-ignore]
    """Wraps ``together.AsyncStream`` to finish the span when the stream is consumed."""

    def __init__(self, stream: Any, with_span: _WithSpan) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_accumulator = _ChunkAccumulator()

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._finish_success()
            raise
        except Exception as exception:
            self._finish_error(exception)
            raise
        self._process_chunk(chunk)
        return chunk

    async def __aenter__(self) -> "_AsyncStream":
        await self.__wrapped__.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        result = await self.__wrapped__.__aexit__(*args)
        self._finish_success()
        return result

    async def close(self) -> None:
        await self.__wrapped__.close()
        self._finish_success()
