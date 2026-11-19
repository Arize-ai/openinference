"""A streamed LLM call must run under its LlamaIndex span, as a non-streamed one does.

`stream_chat` and the other stream methods return their generator before any request is
made: the provider request starts on the first step, after the dispatcher has exited the
span. Each step of the stream must run with the LlamaIndex span current, or the provider
span (here OpenAIInstrumentor's ChatCompletion) starts a trace of its own.
"""

import asyncio
import json
import threading
from itertools import zip_longest
from typing import Any, AsyncIterator, Iterator, List, Tuple, Type

import httpx
import pytest
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.llms import (
    ChatMessage,
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.llms.openai import OpenAI
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from openinference.instrumentation import suppress_tracing
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.llama_index._llm_stream import (
    _ORIGINAL,
    _STREAM_METHODS,
    _asteps_under,
    _steps_under,
)
from openinference.instrumentation.openai import OpenAIInstrumentor

MESSAGES = [ChatMessage(role="user", content="Capital of France?")]
SPAN = NonRecordingSpan(SpanContext(1, 1, is_remote=False, trace_flags=TraceFlags(1)))


def _sse_body() -> bytes:
    def chunk(delta: Any, finish: Any = None) -> str:
        body = {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        return f"data: {json.dumps(body)}\n\n"

    events = [
        chunk({"role": "assistant", "content": ""}),
        chunk({"content": "Par"}),
        chunk({"content": "is."}),
        chunk({}, "stop"),
        "data: [DONE]\n\n",
    ]
    return "".join(events).encode()


def _answer(request: httpx.Request) -> httpx.Response:
    assert json.loads(request.content).get("stream")
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_body())


async def _aanswer(request: httpx.Request) -> httpx.Response:
    return _answer(request)


def _llm(cls: Type[OpenAI] = OpenAI) -> OpenAI:
    return cls(
        model="gpt-4o-mini",
        api_key="sk-test",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(_answer)),
        async_http_client=httpx.AsyncClient(transport=httpx.MockTransport(_aanswer)),
    )


@pytest.fixture
def instrumented(tracer_provider: TracerProvider) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OpenAIInstrumentor().uninstrument()
    LlamaIndexInstrumentor().uninstrument()


def _pairs(exporter: InMemorySpanExporter, name: str) -> List[Tuple[ReadableSpan, ReadableSpan]]:
    """Each ChatCompletion span with its parent, which must be the LlamaIndex `name` span."""
    spans = exporter.get_finished_spans()
    by_id = {span.context.span_id: span for span in spans}
    pairs = []
    for span in spans:
        if span.name != "ChatCompletion":
            continue
        assert span.parent is not None, "the provider span started a trace of its own"
        outer = by_id[span.parent.span_id]
        assert outer.name == name
        assert outer.context.trace_id == span.context.trace_id
        pairs.append((outer, span))
    return pairs


def _nothing_current() -> bool:
    return not trace_api.get_current_span().get_span_context().is_valid


def test_stream_chat_nests(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    chunks = list(_llm().stream_chat(MESSAGES))
    assert chunks[-1].message.content == "Paris."
    assert len(_pairs(in_memory_span_exporter, "OpenAI.stream_chat")) == 1
    assert _nothing_current()


def test_stream_complete_nests(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    chunks = list(_llm().stream_complete("Capital of France?"))
    assert chunks[-1].text == "Paris."
    assert len(_pairs(in_memory_span_exporter, "OpenAI.stream_complete")) == 1
    assert _nothing_current()


async def test_astream_chat_nests(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    chunks = [chunk async for chunk in await _llm().astream_chat(MESSAGES)]
    assert chunks[-1].message.content == "Paris."
    assert len(_pairs(in_memory_span_exporter, "OpenAI.astream_chat")) == 1
    assert _nothing_current()


async def test_astream_complete_nests(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    chunks = [chunk async for chunk in await _llm().astream_complete("Capital of France?")]
    assert chunks[-1].text == "Paris."
    assert len(_pairs(in_memory_span_exporter, "OpenAI.astream_complete")) == 1
    assert _nothing_current()


def test_the_span_is_not_current_between_chunks(instrumented: None) -> None:
    for _ in _llm().stream_chat(MESSAGES):
        assert _nothing_current()


def test_a_span_opened_around_the_stream_stays_the_parent(
    instrumented: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with tracer_provider.get_tracer(__name__).start_as_current_span("caller"):
        stream = _llm().stream_chat(MESSAGES)
        for _ in stream:
            assert trace_api.get_current_span().name == "caller"  # type: ignore[attr-defined]
    ((outer, _),) = _pairs(in_memory_span_exporter, "OpenAI.stream_chat")
    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    assert outer.parent is not None
    assert outer.parent.span_id == spans["caller"].context.span_id


def test_closing_a_stream_early_closes_the_stream_it_wraps() -> None:
    closed: List[str] = []

    def inner() -> Iterator[int]:
        try:
            yield 1
            yield 2
        except GeneratorExit:
            closed.append("sync")
            raise

    # The inner stream stays referenced, so only the wrapper's own close() can close it.
    source = inner()
    stream = _steps_under(SPAN, source)
    assert next(stream) == 1
    stream.close()
    assert closed == ["sync"]
    assert _nothing_current()


async def test_closing_an_async_stream_early_closes_the_stream_it_wraps() -> None:
    closed: List[str] = []

    async def inner() -> AsyncIterator[int]:
        try:
            yield 1
            yield 2
        except GeneratorExit:
            closed.append("async")
            raise

    source = inner()
    stream = _asteps_under(SPAN, source)
    assert await stream.__anext__() == 1
    await stream.aclose()
    assert closed == ["async"]
    assert _nothing_current()


def test_parallel_streams_in_threads_each_nest_under_their_own(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    barrier = threading.Barrier(4)
    errors: List[BaseException] = []

    def read() -> None:
        try:
            stream = _llm().stream_chat(MESSAGES)
            barrier.wait()
            list(stream)
        except BaseException as exception:
            errors.append(exception)

    threads = [threading.Thread(target=read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    pairs = _pairs(in_memory_span_exporter, "OpenAI.stream_chat")
    assert len(pairs) == 4
    assert len({outer.context.span_id for outer, _ in pairs}) == 4


async def test_concurrent_streams_in_tasks_each_nest_under_their_own(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    async def read() -> None:
        async for _ in await _llm().astream_chat(MESSAGES):
            await asyncio.sleep(0)

    await asyncio.gather(*(read() for _ in range(4)))
    pairs = _pairs(in_memory_span_exporter, "OpenAI.astream_chat")
    assert len(pairs) == 4
    assert len({outer.context.span_id for outer, _ in pairs}) == 4


def test_interleaved_streams_each_nest_under_their_own(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    first, second = _llm().stream_chat(MESSAGES), _llm().stream_chat(MESSAGES)
    for a, b in zip_longest(first, second):
        assert a.delta == b.delta
    pairs = _pairs(in_memory_span_exporter, "OpenAI.stream_chat")
    assert len(pairs) == 2
    assert len({outer.context.span_id for outer, _ in pairs}) == 2


def test_an_llm_defined_after_instrumenting_nests(
    instrumented: None,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = tracer_provider.get_tracer(__name__)

    # A stream of its own that calls its provider in the first step, as an integration does.
    class LaterLLM(CustomLLM):
        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata()

        def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
            raise NotImplementedError

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> Any:
            def gen() -> Iterator[CompletionResponse]:
                with tracer.start_as_current_span("provider"):
                    yield CompletionResponse(text="Paris.", delta="Paris.")

            return gen()

    assert hasattr(LaterLLM.__dict__["stream_complete"], _ORIGINAL)
    list(LaterLLM().stream_complete("Capital of France?"))
    spans = {span.name: span for span in in_memory_span_exporter.get_finished_spans()}
    provider, outer = spans["provider"], spans["LaterLLM.stream_complete"]
    assert provider.parent is not None
    assert provider.parent.span_id == outer.context.span_id
    assert _nothing_current()


def test_suppressed_when_called_the_stream_is_not_traced(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with suppress_tracing():
        stream = _llm().stream_chat(MESSAGES)
    assert stream.gi_code.co_name != _steps_under.__name__  # returned as is
    with suppress_tracing():
        list(stream)
    assert not in_memory_span_exporter.get_finished_spans()


def test_suppressed_while_read_the_provider_call_is_not_traced(
    instrumented: None, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    stream = _llm().stream_chat(MESSAGES)
    with suppress_tracing():
        list(stream)
    names = [span.name for span in in_memory_span_exporter.get_finished_spans()]
    assert "ChatCompletion" not in names


def test_uninstrumenting_restores_the_stream_methods(tracer_provider: TracerProvider) -> None:
    def methods(cls: Type[Any]) -> List[Any]:
        return [cls.__dict__.get(name) for name in _STREAM_METHODS]

    before = methods(OpenAI)
    hook = BaseLLM.__dict__.get("__pydantic_init_subclass__")
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    assert methods(OpenAI) != before

    class DefinedWhileInstrumented(OpenAI):
        def stream_chat(self, *args: Any, **kwargs: Any) -> Any:
            return super().stream_chat(*args, **kwargs)

    assert hasattr(DefinedWhileInstrumented.__dict__["stream_chat"], _ORIGINAL)
    LlamaIndexInstrumentor().uninstrument()
    assert methods(OpenAI) == before
    assert not hasattr(DefinedWhileInstrumented.__dict__["stream_chat"], _ORIGINAL)
    assert BaseLLM.__dict__.get("__pydantic_init_subclass__") is hook
