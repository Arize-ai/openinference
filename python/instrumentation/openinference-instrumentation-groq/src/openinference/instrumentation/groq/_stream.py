from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue
from wrapt import ObjectProxy

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.groq._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.instrumentation.groq._utils import _finish_tracing, get_attribute
from openinference.instrumentation.groq._with_span import _WithSpan
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ("_AsyncStream", "_ChatCompletionAccumulator", "_Stream")


class _ChatCompletionAccumulator:
    """Folds ``chat.completion.chunk`` payloads into one chat completion.

    A streaming call returns before any content arrives, so the span is finished
    from the accumulated chunks instead of from the ``Stream`` object itself.
    """

    __slots__ = (
        "_response",
        "_result_cache",
        "_request_parameters",
        "_response_extractor",
        "_values",
    )

    def __init__(
        self,
        request_parameters: Mapping[str, Any],
        response_extractor: _ResponseAttributesExtractor,
    ) -> None:
        self._request_parameters = request_parameters
        self._response_extractor = response_extractor
        self._response: Optional[Any] = None
        self._result_cache: Optional[Dict[str, Any]] = None
        self._values: Dict[str, Any] = {"choices": {}}

    def process_chunk(self, chunk: Any) -> None:
        self._response = None
        self._result_cache = None
        try:
            values = chunk.model_dump(exclude_unset=True)
        except Exception:
            logger.exception(f"Failed to dump stream chunk of type {type(chunk)}")
            return
        if not isinstance(values, Mapping):
            return
        for key in ("id", "created", "model", "system_fingerprint"):
            if (value := values.get(key)) is not None:
                self._values[key] = value
        usage = values.get("usage")
        if usage is None and isinstance(x_groq := values.get("x_groq"), Mapping):
            usage = x_groq.get("usage")
        if usage is not None:
            self._values["usage"] = usage
        for choice in values.get("choices") or ():
            if not isinstance(choice, Mapping):
                continue
            index = choice.get("index")
            if index is None:
                index = 0
            entry = self._values["choices"].setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": None, "content": [], "tool_calls": {}},
                    "finish_reason": None,
                },
            )
            message = entry["message"]
            if (delta := choice.get("delta")) is not None:
                if role := get_attribute(delta, "role"):
                    message["role"] = role
                if (content := get_attribute(delta, "content")) is not None:
                    message["content"].append(content)
                if (reasoning := get_attribute(delta, "reasoning")) is not None:
                    message.setdefault("reasoning", []).append(reasoning)
                for tool_call in get_attribute(delta, "tool_calls") or ():
                    self._process_tool_call(message["tool_calls"], tool_call)
            if (finish_reason := choice.get("finish_reason")) is not None:
                entry["finish_reason"] = finish_reason

    @staticmethod
    def _process_tool_call(tool_calls: Dict[Any, Dict[str, Any]], tool_call: Any) -> None:
        index = get_attribute(tool_call, "index")
        if index is None:
            index = 0
        entry = tool_calls.setdefault(
            index, {"id": None, "type": None, "function": {"name": [], "arguments": []}}
        )
        if tool_call_id := get_attribute(tool_call, "id"):
            entry["id"] = tool_call_id
        if tool_call_type := get_attribute(tool_call, "type"):
            entry["type"] = tool_call_type
        function = get_attribute(tool_call, "function")
        if function is None:
            return
        if name := get_attribute(function, "name"):
            entry["function"]["name"].append(name)
        if arguments := get_attribute(function, "arguments"):
            entry["function"]["arguments"].append(arguments)

    def _result(self) -> Optional[Dict[str, Any]]:
        """The accumulated chunks as a plain chat-completion payload."""
        if (cached := self._result_cache) is not None:
            return cached
        if not self._values["choices"]:
            return None
        result: Dict[str, Any] = {
            "id": self._values.get("id"),
            "object": "chat.completion",
            "created": self._values.get("created"),
            "model": self._values.get("model"),
            "choices": [],
        }
        if (system_fingerprint := self._values.get("system_fingerprint")) is not None:
            result["system_fingerprint"] = system_fingerprint
        if (usage := self._values.get("usage")) is not None:
            result["usage"] = usage
        for index, entry in sorted(self._values["choices"].items()):
            message = entry["message"]
            tool_calls: Optional[List[Dict[str, Any]]] = None
            if message["tool_calls"]:
                tool_calls = [
                    {
                        "id": tool_call["id"],
                        "type": tool_call["type"] or "function",
                        "function": {
                            "name": "".join(tool_call["function"]["name"]),
                            "arguments": "".join(tool_call["function"]["arguments"]),
                        },
                    }
                    for _, tool_call in sorted(message["tool_calls"].items())
                ]
            content = "".join(message["content"])
            result["choices"].append(
                {
                    "index": index,
                    "message": {
                        "role": message["role"] or "assistant",
                        "content": content or None,
                        **(
                            {"reasoning": "".join(message["reasoning"])}
                            if message.get("reasoning")
                            else {}
                        ),
                        **({"tool_calls": tool_calls} if tool_calls else {}),
                    },
                    "finish_reason": entry["finish_reason"],
                }
            )
        self._result_cache = result
        return result

    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if (result := self._result()) is None:
            return
        yield SpanAttributes.OUTPUT_VALUE, safe_json_dumps(result)
        yield SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        if (response := self._build_response()) is None:
            return
        yield from self._response_extractor.get_extra_attributes(
            response=response,
            request_parameters=self._request_parameters,
        )

    def _build_response(self) -> Optional[Any]:
        """A completed ``ChatCompletion`` assembled from the accumulated chunks.

        The response extractor reads the same shape from a streamed call as from
        a regular one. ``model_construct`` is used instead of validation so an
        incomplete stream (no finish reason yet) can still report what arrived.
        """
        if (cached := self._response) is not None:
            return cached
        result = self._result()
        if result is None:
            return None
        try:
            from groq.types.chat.chat_completion import ChatCompletion, Choice
            from groq.types.chat.chat_completion_message import ChatCompletionMessage
            from groq.types.chat.chat_completion_message_tool_call import (
                ChatCompletionMessageToolCall,
                Function,
            )
            from groq.types.completion_usage import CompletionUsage
        except Exception:
            logger.exception("Failed to import the groq chat completion types")
            return None
        try:
            choices = []
            for choice in result["choices"]:
                message = choice["message"]
                message_values: Dict[str, Any] = {
                    "role": message["role"],
                    "content": message["content"],
                }
                if message.get("reasoning") is not None:
                    message_values["reasoning"] = message["reasoning"]
                if message.get("tool_calls"):
                    message_values["tool_calls"] = [
                        ChatCompletionMessageToolCall.model_construct(
                            id=tool_call["id"] or "",
                            type=tool_call["type"] or "function",
                            function=Function.model_construct(
                                name=tool_call["function"]["name"],
                                arguments=tool_call["function"]["arguments"],
                            ),
                        )
                        for tool_call in message["tool_calls"]
                    ]
                choices.append(
                    Choice.model_construct(
                        index=choice["index"],
                        message=ChatCompletionMessage.model_construct(**message_values),
                        finish_reason=choice["finish_reason"],
                        logprobs=None,
                    )
                )
            usage = None
            if (usage_values := result.get("usage")) is not None:
                usage = CompletionUsage.model_construct(**usage_values)
            response = ChatCompletion.model_construct(
                id=result.get("id") or "",
                object="chat.completion",
                created=result.get("created") or 0,
                model=result.get("model") or "",
                choices=choices,
                usage=usage,
                system_fingerprint=result.get("system_fingerprint"),
            )
        except Exception:
            logger.exception("Failed to assemble the streamed chat completion")
            return None
        self._response = response
        return response


class _Stream(ObjectProxy):  # type: ignore[misc,type-arg,unused-ignore]
    """Wraps the sync SDK stream so the span covers consumption, not just the call."""

    __slots__ = (
        "_self_accumulator",
        "_self_is_finished",
        "_self_iteration_count",
        "_self_with_span",
    )

    def __init__(
        self,
        stream: Any,
        with_span: _WithSpan,
        accumulator: _ChatCompletionAccumulator,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_accumulator = accumulator
        self._self_iteration_count = 0
        self._self_is_finished = with_span.is_finished

    def __iter__(self) -> Iterator[Any]:
        return self

    def __next__(self) -> Any:
        try:
            chunk: Any = self.__wrapped__.__next__()
        except StopIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except Exception as exception:
            self._finish_error(exception)
            raise
        self._process_chunk(chunk)
        return chunk

    def __enter__(self) -> Any:
        # The SDK's stream returns itself from __enter__; return the proxy so
        # iteration inside a ``with`` block is still traced.
        obj = self.__wrapped__.__enter__()
        if obj is self.__wrapped__:
            return self
        return obj

    def __exit__(self, *args: Any) -> Any:
        try:
            return self.__wrapped__.__exit__(*args)
        finally:
            # Exited before exhaustion: leave the status UNSET so a truncated
            # stream can still be told apart from a completed one.
            self._finish()

    def close(self) -> None:
        try:
            close = getattr(self.__wrapped__, "close", None)
            if callable(close):
                close()
        finally:
            self._finish()

    def __del__(self) -> None:
        # Abandoned (possibly never iterated): the span must still be ended.
        try:
            self._finish()
        except BaseException:
            pass

    def _process_chunk(self, chunk: Any) -> None:
        if not self._self_iteration_count:
            try:
                self._self_with_span.add_event("First Token Stream Event")
            except Exception:
                logger.exception("Failed to add event to span")
        self._self_iteration_count += 1
        try:
            self._self_accumulator.process_chunk(chunk)
        except Exception:
            logger.exception("Failed to accumulate stream chunk")

    def _finish_error(self, exception: BaseException) -> None:
        self._self_with_span.record_exception(exception)
        self._finish(
            trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def _finish(self, status: Optional[trace_api.Status] = None) -> None:
        if self._self_is_finished:
            return
        _finish_tracing(
            with_span=self._self_with_span,
            attributes=self._self_accumulator.get_attributes(),
            extra_attributes=self._self_accumulator.get_extra_attributes(),
            status=status,
        )
        self._self_is_finished = True


class _AsyncStream(ObjectProxy):  # type: ignore[misc,type-arg,unused-ignore]
    """Wraps the async SDK stream; ``close`` is a coroutine on that stream."""

    __slots__ = (
        "_self_accumulator",
        "_self_is_finished",
        "_self_iteration_count",
        "_self_with_span",
    )

    def __init__(
        self,
        stream: Any,
        with_span: _WithSpan,
        accumulator: _ChatCompletionAccumulator,
    ) -> None:
        super().__init__(stream)
        self._self_with_span = with_span
        self._self_accumulator = accumulator
        self._self_iteration_count = 0
        self._self_is_finished = with_span.is_finished

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk: Any = await self.__wrapped__.__anext__()
        except StopAsyncIteration:
            self._finish(trace_api.Status(status_code=trace_api.StatusCode.OK))
            raise
        except (Exception, asyncio.CancelledError) as exception:
            self._finish_error(exception)
            raise
        self._process_chunk(chunk)
        return chunk

    async def __aenter__(self) -> Any:
        obj = await self.__wrapped__.__aenter__()
        if obj is self.__wrapped__:
            return self
        return obj

    async def __aexit__(self, *args: Any) -> Any:
        try:
            return await self.__wrapped__.__aexit__(*args)
        finally:
            self._finish()

    async def close(self) -> None:
        try:
            close = getattr(self.__wrapped__, "close", None)
            if callable(close):
                await close()
        finally:
            self._finish()

    def __del__(self) -> None:
        # Abandoned (possibly never iterated): the span must still be ended.
        try:
            self._finish()
        except BaseException:
            pass

    def _process_chunk(self, chunk: Any) -> None:
        if not self._self_iteration_count:
            try:
                self._self_with_span.add_event("First Token Stream Event")
            except Exception:
                logger.exception("Failed to add event to span")
        self._self_iteration_count += 1
        try:
            self._self_accumulator.process_chunk(chunk)
        except Exception:
            logger.exception("Failed to accumulate stream chunk")

    def _finish_error(self, exception: BaseException) -> None:
        self._self_with_span.record_exception(exception)
        self._finish(
            trace_api.Status(
                status_code=trace_api.StatusCode.ERROR,
                description=f"{type(exception).__name__}: {exception}",
            )
        )

    def _finish(self, status: Optional[trace_api.Status] = None) -> None:
        if self._self_is_finished:
            return
        _finish_tracing(
            with_span=self._self_with_span,
            attributes=self._self_accumulator.get_attributes(),
            extra_attributes=self._self_accumulator.get_extra_attributes(),
            status=status,
        )
        self._self_is_finished = True
