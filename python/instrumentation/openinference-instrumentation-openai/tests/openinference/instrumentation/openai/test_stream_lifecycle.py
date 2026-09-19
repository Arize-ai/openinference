"""The LLM span must be ended when the caller stops consuming a stream early.

Leaving the context manager, calling `close()` / `aclose()`, or dropping the stream object used to
hand control back without ever raising `StopIteration`, so the span stayed open, was never
exported, and the whole LLM call disappeared from the trace.
"""

import asyncio
import gc
import importlib
from typing import Any, AsyncIterator, Dict, Iterable, Iterator, List, Tuple
from urllib.parse import urljoin

import pytest
from httpx import AsyncByteStream, Response
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from respx import MockRouter

_OPENAI_BASE_URL = "https://api.openai.com/v1/"
_OPENINFERENCE_SCOPE = "openinference.instrumentation.openai"
_MODEL = "gpt-4o-mini"


class _MockAsyncByteStream(AsyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]) -> None:
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string


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


def _mock_stream(respx_mock: MockRouter) -> None:
    url = urljoin(_OPENAI_BASE_URL, "chat/completions")
    respx_mock.post(url).mock(
        return_value=Response(
            status_code=200,
            stream=_MockAsyncByteStream(_sse_events()),
        )
    )


def _client(is_async: bool) -> Any:
    openai = importlib.import_module("openai")
    factory = openai.AsyncOpenAI if is_async else openai.OpenAI
    return factory(api_key="sk-", base_url=_OPENAI_BASE_URL)


# Each test keeps a strong reference to the stream until after its assertions, so a span can only
# be finished by the teardown path the test exercises, not by a later __del__ during GC.


@pytest.mark.parametrize("is_async", [False, True])
def test_leaving_context_without_exhausting_ends_one_span(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _mock_stream(respx_mock)
    client = _client(is_async)
    messages: List[Dict[str, str]] = [{"role": "user", "content": "hi"}]
    keep_alive: List[Any] = []

    async def run_async() -> None:
        stream = await client.chat.completions.create(model=_MODEL, messages=messages, stream=True)
        keep_alive.append(stream)
        async with stream:
            async for _ in stream:
                break

    def run_sync() -> None:
        stream = client.chat.completions.create(model=_MODEL, messages=messages, stream=True)
        keep_alive.append(stream)
        with stream:
            for _ in stream:
                break

    if is_async:
        asyncio.run(run_async())
    else:
        run_sync()

    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.UNSET


@pytest.mark.parametrize("is_async", [False, True])
def test_closing_without_exhausting_ends_one_span(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _mock_stream(respx_mock)
    client = _client(is_async)
    messages: List[Dict[str, str]] = [{"role": "user", "content": "hi"}]
    keep_alive: List[Any] = []

    async def run_async() -> None:
        stream = await client.chat.completions.create(model=_MODEL, messages=messages, stream=True)
        keep_alive.append(stream)
        async for _ in stream:
            break
        await stream.close()

    def run_sync() -> None:
        stream = client.chat.completions.create(model=_MODEL, messages=messages, stream=True)
        keep_alive.append(stream)
        for _ in stream:
            break
        stream.close()

    if is_async:
        asyncio.run(run_async())
    else:
        run_sync()

    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.UNSET


def test_aclosing_without_exhausting_ends_one_span(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """`AsyncStream.aclose` is only present in newer releases of the SDK."""
    openai = importlib.import_module("openai")
    if not hasattr(openai.AsyncStream, "aclose"):
        pytest.skip(f"openai {openai.__version__} has no AsyncStream.aclose")
    _mock_stream(respx_mock)
    client = _client(is_async=True)
    messages: List[Dict[str, str]] = [{"role": "user", "content": "hi"}]
    keep_alive: List[Any] = []

    async def run() -> None:
        stream = await client.chat.completions.create(model=_MODEL, messages=messages, stream=True)
        keep_alive.append(stream)
        async for _ in stream:
            break
        await stream.aclose()

    asyncio.run(run())

    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.UNSET


def test_stream_never_iterated_ends_one_span(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    _mock_stream(respx_mock)
    client = _client(is_async=False)
    messages: List[Dict[str, str]] = [{"role": "user", "content": "hi"}]
    stream = client.chat.completions.create(model=_MODEL, messages=messages, stream=True)
    del stream
    gc.collect()

    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.UNSET


@pytest.mark.parametrize("is_async", [False, True])
def test_exhausted_stream_keeps_ok_status_and_one_span(
    is_async: bool,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Tearing down after full consumption must not re-finish or downgrade the completed span."""
    _mock_stream(respx_mock)
    client = _client(is_async)
    messages: List[Dict[str, str]] = [{"role": "user", "content": "hi"}]
    keep_alive: List[Any] = []

    async def run_async() -> None:
        stream = await client.chat.completions.create(model=_MODEL, messages=messages, stream=True)
        keep_alive.append(stream)
        async with stream:
            async for _ in stream:
                pass
        await stream.close()

    def run_sync() -> None:
        stream = client.chat.completions.create(model=_MODEL, messages=messages, stream=True)
        keep_alive.append(stream)
        with stream:
            for _ in stream:
                pass
        stream.close()

    if is_async:
        asyncio.run(run_async())
    else:
        run_sync()

    (span,) = _llm_spans(in_memory_span_exporter)
    assert span.status.status_code == trace_api.StatusCode.OK
