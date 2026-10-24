"""anthropic>=1.14.0 starts a span of its own for each API call. A call that the instrumentor
traces has its LLM span instead, which the SDK's request continues."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic, NotFoundError
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from wrapt import FunctionWrapper

from openinference.instrumentation.anthropic import AnthropicInstrumentor

pytest.importorskip(
    "anthropic.lib._tracing", reason="this anthropic version starts no spans of its own"
)

MODEL = "claude-sonnet-4-6"
OI_SCOPE = "openinference.instrumentation.anthropic"
SDK_SCOPE = "com.anthropic.sdk.python"
KWARGS: Dict[str, Any] = dict(
    model=MODEL, max_tokens=16, messages=[{"role": "user", "content": "hi"}]
)
MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": MODEL,
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 2},
}
EVENTS: List[Dict[str, Any]] = [
    {
        "type": "message_start",
        "message": {
            **MESSAGE,
            "content": [],
            "stop_reason": None,
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 2},
    },
    {"type": "message_stop"},
]


class _Server:
    """Answers the Messages API, and records what was current and sent when each request went."""

    def __init__(self) -> None:
        self.current_spans: List[trace_api.SpanContext] = []
        self.traceparents: List[Optional[str]] = []
        # called while a Messages request is being sent
        self.on_message_request: Optional[Callable[[], None]] = None

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.current_spans.append(trace_api.get_current_span().get_span_context())
        self.traceparents.append(request.headers.get("traceparent"))
        if request.url.path == "/v1/messages" and self.on_message_request is not None:
            self.on_message_request()
        if request.url.path.endswith("/count_tokens"):
            return httpx2.Response(200, json={"input_tokens": 10})
        if request.url.path.endswith("/models"):
            return httpx2.Response(200, json={"data": [], "has_more": False})
        body = json.loads(request.content)
        if body["model"] == "missing":
            error = {"type": "not_found_error", "message": "model: missing"}
            return httpx2.Response(404, json={"type": "error", "error": error})
        if body.get("stream"):
            content = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in EVENTS)
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"}, content=content.encode()
            )
        return httpx2.Response(200, json=MESSAGE)


def _sdk_spans_to(tracer_provider: TracerProvider) -> Dict[str, Any]:
    # the SDK's spans go to the global tracer provider by default
    return {"open_telemetry": {"tracer_provider": tracer_provider}}


def _no_sdk_spans() -> Dict[str, Any]:
    return {"open_telemetry": False}


@pytest.fixture
def server() -> _Server:
    return _Server()


@pytest.fixture
def client(server: _Server, tracer_provider: TracerProvider) -> Anthropic:
    return Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
        **_sdk_spans_to(tracer_provider),
    )


@pytest.fixture
def async_client(server: _Server, tracer_provider: TracerProvider) -> AsyncAnthropic:
    async def handler(request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        return server(request)

    return AsyncAnthropic(
        api_key="sk-ant-fake",
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
        **_sdk_spans_to(tracer_provider),
    )


def _create(client: Anthropic) -> None:
    client.messages.create(**KWARGS)


def _create_streaming(client: Anthropic) -> None:
    for _ in client.messages.create(**KWARGS, stream=True):
        pass


def _stream(client: Anthropic) -> None:
    with client.messages.stream(**KWARGS) as stream:
        stream.get_final_message()


def _parse(client: Anthropic) -> None:
    client.messages.parse(**KWARGS)


def _count_tokens(client: Anthropic) -> None:
    client.messages.count_tokens(model=MODEL, messages=KWARGS["messages"])


def _raw_response(client: Anthropic) -> None:
    client.messages.with_raw_response.create(**KWARGS).parse()


def _beta_create(client: Anthropic) -> None:
    client.beta.messages.create(**KWARGS)


def _beta_create_streaming(client: Anthropic) -> None:
    for _ in client.beta.messages.create(**KWARGS, stream=True):
        pass


def _beta_stream(client: Anthropic) -> None:
    with client.beta.messages.stream(**KWARGS) as stream:
        stream.get_final_message()


def _beta_count_tokens(client: Anthropic) -> None:
    client.beta.messages.count_tokens(model=MODEL, messages=KWARGS["messages"])


SYNC_CALLS: List[Callable[[Anthropic], None]] = [
    _create,
    _create_streaming,
    _stream,
    _parse,
    _count_tokens,
    _raw_response,
    _beta_create,
    _beta_create_streaming,
    _beta_stream,
    _beta_count_tokens,
]


async def _async_create(client: AsyncAnthropic) -> None:
    await client.messages.create(**KWARGS)


async def _async_create_streaming(client: AsyncAnthropic) -> None:
    async for _ in await client.messages.create(**KWARGS, stream=True):
        pass


async def _async_stream(client: AsyncAnthropic) -> None:
    async with client.messages.stream(**KWARGS) as stream:
        await stream.get_final_message()


async def _async_count_tokens(client: AsyncAnthropic) -> None:
    await client.messages.count_tokens(model=MODEL, messages=KWARGS["messages"])


async def _async_beta_create(client: AsyncAnthropic) -> None:
    await client.beta.messages.create(**KWARGS)


async def _async_beta_stream(client: AsyncAnthropic) -> None:
    async with client.beta.messages.stream(**KWARGS) as stream:
        await stream.get_final_message()


ASYNC_CALLS: List[Callable[[AsyncAnthropic], Awaitable[None]]] = [
    _async_create,
    _async_create_streaming,
    _async_stream,
    _async_count_tokens,
    _async_beta_create,
    _async_beta_stream,
]


def _scopes(exporter: InMemorySpanExporter) -> List[str]:
    return [
        span.instrumentation_scope.name
        for span in _spans(exporter)
        if span.instrumentation_scope is not None
    ]


def _spans(exporter: InMemorySpanExporter) -> List[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.instrumentation_scope is not None and span.instrumentation_scope.name != "test"
    ]


def _assert_request_continues(server: _Server, span: ReadableSpan) -> None:
    """The request went while the LLM span was current, and carried its trace context."""
    context = span.get_span_context()
    assert context is not None
    assert server.current_spans == [context]
    flags = f"{int(context.trace_flags):02x}"
    assert server.traceparents == [f"00-{context.trace_id:032x}-{context.span_id:016x}-{flags}"]


@pytest.mark.parametrize("call", SYNC_CALLS, ids=lambda call: call.__name__.lstrip("_"))
def test_one_span_per_call(
    call: Callable[[Anthropic], None],
    client: Anthropic,
    server: _Server,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    call(client)
    assert _scopes(in_memory_span_exporter) == [OI_SCOPE]
    (span,) = _spans(in_memory_span_exporter)
    _assert_request_continues(server, span)


@pytest.mark.parametrize("call", ASYNC_CALLS, ids=lambda call: call.__name__.lstrip("_"))
async def test_one_span_per_async_call(
    call: Callable[[AsyncAnthropic], Awaitable[None]],
    async_client: AsyncAnthropic,
    server: _Server,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    await call(async_client)
    assert _scopes(in_memory_span_exporter) == [OI_SCOPE]
    (span,) = _spans(in_memory_span_exporter)
    _assert_request_continues(server, span)


@pytest.mark.parametrize("call", [_create, _stream], ids=["create", "stream"])
def test_llm_span_is_a_child_of_the_callers_span(
    call: Callable[[Anthropic], None],
    client: Anthropic,
    server: _Server,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    with tracer_provider.get_tracer("test").start_as_current_span("parent") as parent:
        call(client)
    (span,) = _spans(in_memory_span_exporter)
    assert span.parent is not None
    assert span.parent.span_id == parent.get_span_context().span_id
    _assert_request_continues(server, span)


def test_failed_call_has_one_span(
    client: Anthropic,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    with pytest.raises(NotFoundError):
        client.messages.create(**{**KWARGS, "model": "missing"})
    (span,) = _spans(in_memory_span_exporter)
    assert span.instrumentation_scope is not None
    assert span.instrumentation_scope.name == OI_SCOPE
    assert span.status.status_code == trace_api.StatusCode.ERROR


def _traceparent(span: ReadableSpan) -> str:
    context = span.get_span_context()
    assert context is not None
    return f"00-{context.trace_id:032x}-{context.span_id:016x}-{int(context.trace_flags):02x}"


def test_concurrent_calls_in_threads(
    client: Anthropic,
    server: _Server,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    calls = [_create, _stream, _create_streaming, _count_tokens] * 4
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda call: call(client), calls))
    spans = _spans(in_memory_span_exporter)
    assert _scopes(in_memory_span_exporter) == [OI_SCOPE] * len(calls)
    assert sorted(map(str, server.traceparents)) == sorted(_traceparent(span) for span in spans)


async def test_concurrent_async_calls(
    async_client: AsyncAnthropic,
    server: _Server,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    calls = [_async_create, _async_stream, _async_create_streaming, _async_count_tokens] * 4
    await asyncio.gather(*(call(async_client) for call in calls))
    spans = _spans(in_memory_span_exporter)
    assert _scopes(in_memory_span_exporter) == [OI_SCOPE] * len(calls)
    assert sorted(map(str, server.traceparents)) == sorted(_traceparent(span) for span in spans)


@pytest.mark.parametrize("own_span", [False, True], ids=["without_span", "with_span"])
def test_another_call_made_while_the_request_is_sent_keeps_the_sdk_span(
    own_span: bool,
    client: Anthropic,
    server: _Server,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    def list_models() -> None:
        server.on_message_request = None
        if own_span:
            with tracer_provider.get_tracer("test").start_as_current_span("middleware"):
                client.models.list()
        else:
            client.models.list()

    server.on_message_request = list_models
    client.messages.create(**KWARGS)
    spans = {
        span.instrumentation_scope.name: span
        for span in _spans(in_memory_span_exporter)
        if span.instrumentation_scope is not None
    }
    assert sorted(spans) == [SDK_SCOPE, OI_SCOPE]
    sdk_span, llm_span = spans[SDK_SCOPE], spans[OI_SCOPE]
    assert sdk_span.parent is not None
    middlewares = [
        span for span in in_memory_span_exporter.get_finished_spans() if span.name == "middleware"
    ]
    assert len(middlewares) == (1 if own_span else 0)
    parent = middlewares[0] if own_span else llm_span
    context = parent.get_span_context()
    assert context is not None
    assert sdk_span.parent.span_id == context.span_id


def test_sdk_span_stays_when_the_llm_span_is_invalid(
    client: Anthropic,
    server: _Server,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """The instrumentor's tracer provider makes no spans, and the client's does."""
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=trace_api.NoOpTracerProvider())
    try:
        _stream(client)
    finally:
        instrumentor.uninstrument()
    (span,) = _spans(in_memory_span_exporter)
    assert span.instrumentation_scope is not None
    assert span.instrumentation_scope.name == SDK_SCOPE
    assert server.traceparents == [_traceparent(span)]


def test_calls_that_are_not_traced_keep_the_sdk_span(
    client: Anthropic,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    client.models.list()
    assert _scopes(in_memory_span_exporter) == [SDK_SCOPE]


def test_uninstrument_restores_the_sdk_span(
    client: Anthropic,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
) -> None:
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    instrumentor.uninstrument()
    client.messages.create(**KWARGS)
    assert _scopes(in_memory_span_exporter) == [SDK_SCOPE]


def test_sdk_spans_disabled(
    server: _Server,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: None,
) -> None:
    client = Anthropic(
        api_key="sk-ant-fake",
        http_client=httpx2.Client(transport=httpx2.MockTransport(server)),
        **_no_sdk_spans(),
    )
    _stream(client)
    assert _scopes(in_memory_span_exporter) == [OI_SCOPE]


@pytest.fixture
def without_sdk_span_start(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An anthropic version whose client starts its spans some other way."""
    from anthropic import _base_client

    monkeypatch.delattr(_base_client, "start_api_call_span")
    yield


def test_instruments_without_the_sdk_span_start(
    without_sdk_span_start: None,
    tracer_provider: TracerProvider,
) -> None:
    from anthropic.resources.messages import Messages

    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        assert isinstance(Messages.__dict__["create"], FunctionWrapper)
    finally:
        instrumentor.uninstrument()
    assert not isinstance(Messages.__dict__["create"], FunctionWrapper)
