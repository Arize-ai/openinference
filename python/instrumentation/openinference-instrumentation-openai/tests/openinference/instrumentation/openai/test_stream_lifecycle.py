"""The LLM span must be ended when the caller stops consuming a stream early.

Leaving the context manager, calling `close()` / `aclose()`, or dropping the stream object used to
hand control back without ever raising `StopIteration`, so the span stayed open, was never
exported, and the whole LLM call disappeared from the trace.
"""

import asyncio
import gc
import importlib
import json
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    cast,
)
from urllib.parse import urljoin

import pytest
from httpx import AsyncByteStream, Response
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respx import MockRouter

from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_OPENAI_BASE_URL = "https://api.openai.com/v1/"
_OPENINFERENCE_SCOPE = "openinference.instrumentation.openai"
_MODEL = "gpt-4o-mini"
_MESSAGES: List[Dict[str, str]] = [{"role": "user", "content": "hi"}]
_FIRST_TOKEN_EVENT = "First Token Stream Event"
# The role chunk and the first content chunk, so a partial stream has output to report.
_PARTIAL = 2


class _MockAsyncByteStream(AsyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]) -> None:
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string


class _InterruptedByteStream(AsyncByteStream):
    """Sends part of the response, then is interrupted (sync) or never sends the rest (async)."""

    def __iter__(self) -> Iterator[bytes]:
        yield from _sse_events()[:_PARTIAL]
        raise KeyboardInterrupt

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in _sse_events()[:_PARTIAL]:
            yield byte_string
        await asyncio.Event().wait()


def _sse_events() -> List[bytes]:
    return [
        b'data: {"choices": [{"delta": {"role": "assistant"}, "index": 0}]}\n\n',  # noqa: E501
        b'data: {"choices": [{"delta": {"content": "the"}, "index": 0}]}\n\n',  # noqa: E501
        b'data: {"choices": [{"delta": {"content": " answer"}, "index": 0}]}\n\n',  # noqa: E501
        b'data: {"choices": [{"delta": {}, "finish_reason": "stop", "index": 0}]}\n\n',  # noqa: E501
        b"data: [DONE]\n",
    ]


def _llm_spans(exporter: InMemorySpanExporter) -> Tuple[ReadableSpan, ...]:
    return tuple(
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope is not None
        and span.instrumentation_scope.name == _OPENINFERENCE_SCOPE
    )


def _mock_stream(respx_mock: MockRouter, stream: Optional[AsyncByteStream] = None) -> None:
    url = urljoin(_OPENAI_BASE_URL, "chat/completions")
    respx_mock.post(url).mock(
        return_value=Response(
            status_code=200,
            stream=stream or _MockAsyncByteStream(_sse_events()),
        )
    )


def _client(is_async: bool) -> Any:
    openai = importlib.import_module("openai")
    factory = openai.AsyncOpenAI if is_async else openai.OpenAI
    return factory(api_key="sk-", base_url=_OPENAI_BASE_URL)


def _create(client: Any) -> Any:
    return client.chat.completions.create(model=_MODEL, messages=_MESSAGES, stream=True)


def _read(stream: Any, count: Optional[int] = None) -> None:
    for i, _ in enumerate(stream, 1):
        if i == count:
            break


async def _aread(stream: Any, count: Optional[int] = None) -> None:
    i = 0
    async for _ in stream:
        i += 1
        if i == count:
            break


def _assert_attributes(span: ReadableSpan, output: Optional[Dict[str, Any]]) -> None:
    """Checks every attribute on the span; `output` is the message streamed so far, if any."""
    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "openai"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "openai"
    assert json.loads(cast(str, attributes.pop(SpanAttributes.INPUT_VALUE))) == {
        "messages": _MESSAGES,
        "model": _MODEL,
        "stream": True,
    }
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(cast(str, attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS))) == {
        "model": _MODEL,
        "stream": True,
    }
    input_message = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert attributes.pop(f"{input_message}.{MessageAttributes.MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{input_message}.{MessageAttributes.MESSAGE_CONTENT}") == "hi"
    if output is not None:
        assert json.loads(cast(str, attributes.pop(SpanAttributes.OUTPUT_VALUE))) == {
            "choices": [{**output, "index": 0}]
        }
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == (
            OpenInferenceMimeTypeValues.JSON.value
        )
        message = output["message"]
        output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
        assert (
            attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_ROLE}")
            == (message["role"])
        )
        assert (
            attributes.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}")
            == (message["content"])
        )
        if finish_reason := output.get("finish_reason"):
            assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == finish_reason
    assert attributes == {}


_PARTIAL_OUTPUT: Dict[str, Any] = {"message": {"content": "the", "role": "assistant"}}
_FULL_OUTPUT: Dict[str, Any] = {
    "message": {"content": "the answer", "role": "assistant"},
    "finish_reason": "stop",
}

_TEARDOWNS = [
    (False, "context"),
    (False, "close"),
    (True, "context"),
    (True, "close"),
    (True, "aclose"),
]


# Each test keeps a strong reference to the stream until after its assertions, so a span can only
# be finished by the teardown path the test exercises, not by a later __del__ during GC.


@pytest.mark.parametrize("is_async,teardown", _TEARDOWNS)
def test_early_teardown_ends_one_span_with_partial_output(
    is_async: bool,
    teardown: str,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _mock_stream(respx_mock)
    client = _client(is_async)
    keep_alive: List[Any] = []

    async def run_async() -> None:
        stream = await _create(client)
        keep_alive.append(stream)
        if teardown == "context":
            async with stream:
                await _aread(stream, _PARTIAL)
            return
        await _aread(stream, _PARTIAL)
        # Older releases of the SDK have no `AsyncStream.aclose`, so `close` is used instead.
        await (stream.aclose() if teardown == "aclose" else stream.close())

    def run_sync() -> None:
        stream = _create(client)
        keep_alive.append(stream)
        if teardown == "context":
            with stream:
                _read(stream, _PARTIAL)
            return
        _read(stream, _PARTIAL)
        stream.close()

    if is_async:
        asyncio.run(run_async())
    else:
        run_sync()

    assert keep_alive[0].response.is_closed
    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.UNSET
    assert [event.name for event in span.events] == [_FIRST_TOKEN_EVENT]
    _assert_attributes(span, _PARTIAL_OUTPUT)


@pytest.mark.parametrize(
    "is_async,teardown,exception_type",
    [
        (is_async, teardown, exception_type)
        for is_async, teardown in _TEARDOWNS
        # A cancelled or interrupted teardown is reported the same way as one that fails.
        for exception_type in (
            RuntimeError,
            asyncio.CancelledError if is_async else KeyboardInterrupt,
        )
    ],
)
def test_teardown_that_raises_ends_one_span_with_error(
    is_async: bool,
    teardown: str,
    exception_type: Type[BaseException],
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_stream(respx_mock)
    client = _client(is_async)
    keep_alive: List[Any] = []

    def close() -> None:
        raise exception_type("close failed")

    async def aclose() -> None:
        raise exception_type("close failed")

    async def run_async() -> None:
        stream = await _create(client)
        keep_alive.append(stream)
        monkeypatch.setattr(stream.response, "aclose", aclose)
        if teardown == "context":
            async with stream:
                await _aread(stream, _PARTIAL)
            return
        await _aread(stream, _PARTIAL)
        await (stream.aclose() if teardown == "aclose" else stream.close())

    def run_sync() -> None:
        stream = _create(client)
        keep_alive.append(stream)
        monkeypatch.setattr(stream.response, "close", close)
        if teardown == "context":
            with stream:
                _read(stream, _PARTIAL)
            return
        _read(stream, _PARTIAL)
        stream.close()

    # No message match: asyncio.run drops the message of a CancelledError on Python 3.10.
    with pytest.raises(exception_type):
        if is_async:
            asyncio.run(run_async())
        else:
            run_sync()

    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.status.description == f"{exception_type.__name__}: close failed"
    assert [event.name for event in span.events] == [_FIRST_TOKEN_EVENT, "exception"]
    exception_event = span.events[1]
    module = exception_type.__module__
    assert dict(exception_event.attributes or {})["exception.type"] == (
        exception_type.__name__ if module == "builtins" else f"{module}.{exception_type.__name__}"
    )
    assert dict(exception_event.attributes or {})["exception.message"] == "close failed"
    _assert_attributes(span, _PARTIAL_OUTPUT)


@pytest.mark.parametrize(
    "is_async,exception_type",
    [
        (False, RuntimeError),
        (False, KeyboardInterrupt),
        (True, RuntimeError),
        (True, asyncio.CancelledError),
    ],
)
def test_context_body_that_raises_ends_one_span_with_error(
    is_async: bool,
    exception_type: Type[BaseException],
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The teardown itself succeeds, so the error only reaches the stream through its exit."""
    _mock_stream(respx_mock)
    client = _client(is_async)
    keep_alive: List[Any] = []

    async def run_async() -> None:
        stream = await _create(client)
        keep_alive.append(stream)
        async with stream:
            await _aread(stream, _PARTIAL)
            raise exception_type("body failed")

    def run_sync() -> None:
        stream = _create(client)
        keep_alive.append(stream)
        with stream:
            _read(stream, _PARTIAL)
            raise exception_type("body failed")

    # No message match: asyncio.run drops the message of a CancelledError on Python 3.10.
    with pytest.raises(exception_type):
        if is_async:
            asyncio.run(run_async())
        else:
            run_sync()

    assert keep_alive[0].response.is_closed
    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.status.description == f"{exception_type.__name__}: body failed"
    assert [event.name for event in span.events] == [_FIRST_TOKEN_EVENT, "exception"]
    exception_event = span.events[1]
    module = exception_type.__module__
    assert dict(exception_event.attributes or {})["exception.type"] == (
        exception_type.__name__ if module == "builtins" else f"{module}.{exception_type.__name__}"
    )
    assert dict(exception_event.attributes or {})["exception.message"] == "body failed"
    _assert_attributes(span, _PARTIAL_OUTPUT)


def test_closing_a_generator_that_holds_the_context_ends_one_span_without_error(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _mock_stream(respx_mock)
    client = _client(is_async=False)
    keep_alive: List[Any] = []

    def chunks() -> Generator[Any, None, None]:
        stream = _create(client)
        keep_alive.append(stream)
        with stream:
            yield from stream

    generator = chunks()
    for _ in range(_PARTIAL):
        next(generator)
    generator.close()

    assert keep_alive[0].response.is_closed
    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.UNSET
    assert [event.name for event in span.events] == [_FIRST_TOKEN_EVENT]
    _assert_attributes(span, _PARTIAL_OUTPUT)


@pytest.mark.parametrize("is_async", [False, True])
def test_iteration_that_is_cancelled_ends_one_span_with_error(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Cancelling or interrupting while waiting for the next chunk, with no teardown after it."""
    _mock_stream(respx_mock, _InterruptedByteStream())
    client = _client(is_async)
    keep_alive: List[Any] = []
    chunks: List[Any] = []

    async def consume() -> None:
        stream = await _create(client)
        keep_alive.append(stream)
        async for chunk in stream:
            chunks.append(chunk)

    async def run_async() -> None:
        task = asyncio.create_task(consume())
        while len(chunks) < _PARTIAL:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    def run_sync() -> None:
        stream = _create(client)
        keep_alive.append(stream)
        with pytest.raises(KeyboardInterrupt):
            _read(stream)

    if is_async:
        asyncio.run(run_async())
    else:
        run_sync()

    exception_type = "asyncio.exceptions.CancelledError" if is_async else "KeyboardInterrupt"
    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.status.description == f"{exception_type.rpartition('.')[2]}: "
    assert [event.name for event in span.events] == [_FIRST_TOKEN_EVENT, "exception"]
    assert dict(span.events[1].attributes or {})["exception.type"] == exception_type
    _assert_attributes(span, _PARTIAL_OUTPUT)


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("count", [0, _PARTIAL])
def test_dropped_stream_ends_one_span(
    is_async: bool,
    count: int,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """A stream that is dropped without any teardown, whether or not it was iterated."""
    _mock_stream(respx_mock)
    client = _client(is_async)

    async def run_async() -> None:
        stream = await _create(client)
        if count:
            await _aread(stream, count)
        del stream
        gc.collect()

    def run_sync() -> None:
        stream = _create(client)
        if count:
            _read(stream, count)
        del stream
        gc.collect()

    if is_async:
        asyncio.run(run_async())
    else:
        run_sync()

    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.UNSET
    assert [event.name for event in span.events] == ([_FIRST_TOKEN_EVENT] if count else [])
    _assert_attributes(span, _PARTIAL_OUTPUT if count else None)


@pytest.mark.parametrize("is_async", [False, True])
def test_exhausted_stream_keeps_ok_status_and_one_span(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Tearing down after full consumption must not re-finish or downgrade the completed span."""
    _mock_stream(respx_mock)
    client = _client(is_async)
    keep_alive: List[Any] = []

    async def run_async() -> None:
        stream = await _create(client)
        keep_alive.append(stream)
        async with stream:
            await _aread(stream)
        await stream.close()

    def run_sync() -> None:
        stream = _create(client)
        keep_alive.append(stream)
        with stream:
            _read(stream)
        stream.close()

    if is_async:
        asyncio.run(run_async())
    else:
        run_sync()

    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.OK
    assert [event.name for event in span.events] == [_FIRST_TOKEN_EVENT]
    _assert_attributes(span, _FULL_OUTPUT)
