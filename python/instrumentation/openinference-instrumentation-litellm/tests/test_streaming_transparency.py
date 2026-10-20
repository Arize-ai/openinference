import asyncio
import gc
from types import SimpleNamespace
from typing import Any, Dict, Generator, List
from unittest.mock import MagicMock, patch

import litellm
import pytest
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.responses.streaming_iterator import (
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation.litellm import (
    LiteLLMInstrumentor,
    _finalize_sync_streaming_span,
    _instrument_func_type_completion,
    _instrument_func_type_responses,
    _TracedSyncStream,
)
from openinference.semconv.trace import MessageAttributes, SpanAttributes


@pytest.fixture()
def setup_litellm_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LiteLLMInstrumentor().uninstrument()


def test_sync_streaming_preserves_stream_type(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
        mock_response="The capital of China is Beijing",
        stream=True,
    )

    assert isinstance(response, CustomStreamWrapper)

    output_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            output_message += chunk.choices[0].delta.content
    assert output_message == "The capital of China is Beijing"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "completion"


def test_async_streaming_preserves_stream_type(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    async def run() -> str:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
            mock_response="The capital of China is Beijing",
            stream=True,
        )
        assert isinstance(response, CustomStreamWrapper)
        output_message = ""
        async for chunk in response:
            if chunk.choices[0].delta.content:
                output_message += chunk.choices[0].delta.content
        return output_message

    assert asyncio.run(run()) == "The capital of China is Beijing"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "acompletion"


def test_sync_streaming_early_close_records_partial_output(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
        mock_response="The capital of China is Beijing",
        stream=True,
    )

    partial_output = ""
    for chunk in response:
        if content := chunk.choices[0].delta.content:
            partial_output += content
            break
    response.close()

    assert partial_output
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert not spans[0].events
    attributes = dict(spans[0].attributes or {})
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == partial_output
    assert (
        attributes.get(
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
        )
        == partial_output
    )


async def test_async_streaming_early_close_records_partial_output(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    response = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
        mock_response="The capital of China is Beijing",
        stream=True,
    )

    chunk = await response.__anext__()
    partial_output = chunk.choices[0].delta.content
    assert partial_output
    await response.aclose()

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.UNSET
    assert not spans[0].events
    attributes = dict(spans[0].attributes or {})
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == partial_output
    assert (
        attributes.get(
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
        )
        == partial_output
    )


def test_responses_foreign_stream_type_passes_through(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    class ForeignIterator:
        def __init__(self) -> None:
            self._tokens = ["a", "b"]

        def __iter__(self) -> Any:
            return self

        def __next__(self) -> Any:
            if not self._tokens:
                raise StopIteration
            return self._tokens.pop(0)

    foreign = ForeignIterator()
    original_func = LiteLLMInstrumentor.original_litellm_funcs["responses"]

    try:

        def fake_responses(*args: Any, **kwargs: Any) -> Any:
            return foreign

        LiteLLMInstrumentor.original_litellm_funcs["responses"] = fake_responses

        result = litellm.responses(
            model="gpt-4o-mini",
            input="Hi",
            stream=True,
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["responses"] = original_func

    assert result is foreign
    assert list(result) == ["a", "b"]

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "responses"


def test_aresponses_foreign_stream_type_passes_through(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    class ForeignAsyncIterator:
        def __init__(self) -> None:
            self._tokens = ["a", "b"]

        def __aiter__(self) -> Any:
            return self

        async def __anext__(self) -> Any:
            if not self._tokens:
                raise StopAsyncIteration
            return self._tokens.pop(0)

    foreign = ForeignAsyncIterator()
    original_func = LiteLLMInstrumentor.original_litellm_funcs["aresponses"]
    try:

        async def fake_aresponses(*args: Any, **kwargs: Any) -> Any:
            return foreign

        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = fake_aresponses

        async def run() -> Any:
            return await litellm.aresponses(
                model="gpt-4o-mini",
                input="Hi",
                stream=True,
            )

        result = asyncio.run(run())
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = original_func

    assert result is foreign

    async def drain() -> list[Any]:
        return [token async for token in result]

    assert asyncio.run(drain()) == ["a", "b"]

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "aresponses"


def test_attribute_extraction_errors_do_not_raise() -> None:
    cyclic: Dict[str, Any] = {"model": "gpt-4o"}
    cyclic["self"] = cyclic
    cyclic["messages"] = [{"role": "user", "content": "hi"}]

    span = MagicMock()
    assert _instrument_func_type_completion(span, cyclic) is None
    assert _instrument_func_type_responses(span, cyclic) is None


async def test_async_early_close_closes_underlying_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    # Any: the proxy's __wrapped__ is not on litellm's declared return type.
    response: Any = await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
        mock_response="The capital of China is Beijing",
        stream=True,
    )
    await response.__anext__()
    assert response.__wrapped__.completion_stream is not None

    await response.aclose()

    # litellm clears completion_stream in its own aclose to release the
    # provider connection; the instrumented stream must not skip it.
    assert response.__wrapped__.completion_stream is None

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "acompletion"


def test_sync_close_closes_wrapped_stream() -> None:
    # The proxy's close must reach the wrapped stream: litellm's
    # CustomStreamWrapper.close releases the provider connection, and the
    # async proxy already forwards aclose the same way.
    class Wrapped:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    wrapped = Wrapped()
    span = MagicMock()
    proxy = _TracedSyncStream(wrapped, _finalize_sync_streaming_span(span, iter([])), span)
    proxy.close()
    assert wrapped.closed


def test_sync_close_tolerates_wrapped_without_close() -> None:
    # Older litellm versions have no close on the wrapped stream.
    span = MagicMock()
    proxy = _TracedSyncStream(object(), _finalize_sync_streaming_span(span, iter([])), span)
    proxy.close()


def test_sync_early_close_closes_underlying_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    response: Any = litellm.completion(
        model="gpt-3.5-turbo",
        messages=[{"content": "What's the capital of China?", "role": "user"}],
        mock_response="The capital of China is Beijing",
        stream=True,
    )
    next(response)

    with patch.object(CustomStreamWrapper, "close", create=True) as wrapped_close:
        response.close()

    wrapped_close.assert_called_once_with()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "completion"
    assert spans[0].status.status_code == StatusCode.UNSET
    assert not spans[0].events


_MESSAGES = [{"content": "What's the capital of China?", "role": "user"}]


@pytest.mark.parametrize("error", [RuntimeError("provider 429"), asyncio.CancelledError()])
async def test_failed_streaming_call_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    error: BaseException,
) -> None:
    in_memory_span_exporter.clear()
    original_func = LiteLLMInstrumentor.original_litellm_funcs["acompletion"]

    async def failing_acompletion(*args: Any, **kwargs: Any) -> Any:
        raise error

    LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = failing_acompletion
    try:
        with pytest.raises(type(error)):
            await litellm.acompletion(model="gpt-3.5-turbo", messages=_MESSAGES, stream=True)
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert [event.name for event in spans[0].events] == ["exception"]


async def test_async_stream_discarded_unread_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    response = await litellm.acompletion(
        model="gpt-3.5-turbo", messages=_MESSAGES, mock_response="Beijing", stream=True
    )
    assert not in_memory_span_exporter.get_finished_spans()
    del response
    gc.collect()

    assert len(in_memory_span_exporter.get_finished_spans()) == 1


def test_sync_stream_discarded_unread_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    response = litellm.completion(
        model="gpt-3.5-turbo", messages=_MESSAGES, mock_response="Beijing", stream=True
    )
    del response
    gc.collect()

    assert len(in_memory_span_exporter.get_finished_spans()) == 1


async def test_async_stream_closed_unread_ends_span_once(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    response: Any = await litellm.acompletion(
        model="gpt-3.5-turbo", messages=_MESSAGES, mock_response="Beijing", stream=True
    )
    await response.aclose()
    assert response.__wrapped__.completion_stream is None
    del response
    gc.collect()

    assert len(in_memory_span_exporter.get_finished_spans()) == 1


class _FakeResponsesStream(ResponsesAPIStreamingIterator):
    def __init__(self, events: List[Any]) -> None:
        self._events = list(events)

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if not self._events:
            raise StopAsyncIteration
        event = self._events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


async def _drain_aresponses(events: List[Any]) -> None:
    original_func = LiteLLMInstrumentor.original_litellm_funcs["aresponses"]

    async def fake_aresponses(*args: Any, **kwargs: Any) -> Any:
        return _FakeResponsesStream(events)

    LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = fake_aresponses
    try:
        response = await litellm.aresponses(model="gpt-4o-mini", input="Hi", stream=True)
        async for _ in response:
            pass
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = original_func


@pytest.mark.parametrize(
    "event_type, error, status_code",
    [
        (ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE, None, StatusCode.OK),
        (ResponsesAPIStreamEvents.RESPONSE_FAILED, {"message": "boom"}, StatusCode.ERROR),
    ],
)
async def test_aresponses_non_completed_terminal_event_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    event_type: Any,
    error: Any,
    status_code: StatusCode,
) -> None:
    in_memory_span_exporter.clear()

    await _drain_aresponses(
        [SimpleNamespace(type=event_type, response=SimpleNamespace(error=error, usage=None))]
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == status_code


async def test_aresponses_stream_error_ends_span(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    with pytest.raises(RuntimeError, match="stream disconnected"):
        await _drain_aresponses(
            [
                SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA),
                RuntimeError("stream disconnected"),
            ]
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR


class _BlockingResponsesStream(ResponsesAPIStreamingIterator):
    def __init__(self) -> None:
        self._sent = False
        self.blocked = asyncio.Event()

    def __aiter__(self) -> Any:
        return self

    async def __anext__(self) -> Any:
        if not self._sent:
            self._sent = True
            return SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA)
        self.blocked.set()
        await asyncio.Event().wait()


class _FakeSyncResponsesStream(SyncResponsesAPIStreamingIterator):
    def __init__(self, events: List[Any]) -> None:
        self._events = list(events)

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        if not self._events:
            raise StopIteration
        event = self._events.pop(0)
        if isinstance(event, BaseException):
            raise event
        return event


async def test_aresponses_stream_cancelled_mid_stream_ends_span_with_error(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    stream = _BlockingResponsesStream()
    original_func = LiteLLMInstrumentor.original_litellm_funcs["aresponses"]

    async def fake_aresponses(*args: Any, **kwargs: Any) -> Any:
        return stream

    LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = fake_aresponses
    try:
        response = await litellm.aresponses(model="gpt-4o-mini", input="Hi", stream=True)

        async def consume() -> None:
            async for _ in response:
                pass

        task = asyncio.create_task(consume())
        await stream.blocked.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = original_func

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]
    assert span.events[0].attributes is not None
    cancelled = asyncio.CancelledError
    assert span.events[0].attributes["exception.type"] == (
        f"{cancelled.__module__}.{cancelled.__qualname__}"
    )


def test_responses_stream_interrupted_mid_stream_ends_span_with_error(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    interrupt = KeyboardInterrupt()
    original_func = LiteLLMInstrumentor.original_litellm_funcs["responses"]

    def fake_responses(*args: Any, **kwargs: Any) -> Any:
        return _FakeSyncResponsesStream(
            [SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA), interrupt]
        )

    LiteLLMInstrumentor.original_litellm_funcs["responses"] = fake_responses
    try:
        response = litellm.responses(model="gpt-4o-mini", input="Hi", stream=True)
        with pytest.raises(KeyboardInterrupt) as excinfo:
            for _ in response:
                pass
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["responses"] = original_func

    assert excinfo.value is interrupt
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]
    assert span.events[0].attributes is not None
    assert span.events[0].attributes["exception.type"] == "KeyboardInterrupt"


async def test_aresponses_stream_closed_early_ends_span_without_error(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    original_func = LiteLLMInstrumentor.original_litellm_funcs["aresponses"]

    async def fake_aresponses(*args: Any, **kwargs: Any) -> Any:
        return _FakeResponsesStream(
            [
                SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA),
                SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA),
            ]
        )

    LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = fake_aresponses
    try:
        response = await litellm.aresponses(model="gpt-4o-mini", input="Hi", stream=True)
        await response.__anext__()
        await response.aclose()
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["aresponses"] = original_func

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    assert not span.events


def test_responses_stream_closed_early_ends_span_without_error(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    original_func = LiteLLMInstrumentor.original_litellm_funcs["responses"]

    def fake_responses(*args: Any, **kwargs: Any) -> Any:
        return _FakeSyncResponsesStream(
            [
                SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA),
                SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA),
            ]
        )

    LiteLLMInstrumentor.original_litellm_funcs["responses"] = fake_responses
    try:
        response = litellm.responses(model="gpt-4o-mini", input="Hi", stream=True)
        next(response)
        response.close()
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["responses"] = original_func

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.UNSET
    assert not span.events
