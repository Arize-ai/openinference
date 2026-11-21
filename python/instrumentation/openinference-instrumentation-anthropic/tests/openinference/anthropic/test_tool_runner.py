"""Tests for the tracing of beta.messages.tool_runner, sync and async."""

import asyncio
import gc
import json
from typing import Any, Dict, Generator, Iterator, List, Mapping, Optional

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib.tools import beta_async_tool, beta_tool
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation import TraceConfig, using_attributes
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

MODEL = "claude-sonnet-4-6"
SECRET = "SECRET-LOCATION"
WEATHER = "65 degrees and sunny"
ANSWER = "The weather is 65 degrees and sunny."
SCHEMA = {
    "additionalProperties": False,
    "properties": {"location": {"title": "Location", "type": "string"}},
    "required": ["location"],
    "type": "object",
}
TOOL_DEFINITION = {
    "name": "get_weather",
    "description": "Get the weather for a location.",
    "input_schema": SCHEMA,
}
RUNNER_SPAN_NAME = "beta.messages.tool_runner"


@beta_tool
def get_weather(location: str) -> str:
    """Get the weather for a location."""
    return WEATHER


@beta_async_tool(name="get_weather")
async def get_weather_async(location: str) -> str:
    """Get the weather for a location."""
    return WEATHER


@beta_tool
def failing_weather(location: str) -> str:
    """Get the weather for a location."""
    raise RuntimeError("no weather today")


def _message(message_id: str, content: List[Dict[str, Any]], stop_reason: str) -> Dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": MODEL,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _tool_use_message(name: str = "get_weather") -> Dict[str, Any]:
    tool_use = {"type": "tool_use", "id": "toolu_01", "name": name, "input": {"location": SECRET}}
    return _message("msg_01", [tool_use], "tool_use")


def _answer_message() -> Dict[str, Any]:
    return _message("msg_02", [{"type": "text", "text": ANSWER}], "end_turn")


def _server_sent_events(message: Dict[str, Any]) -> bytes:
    events: List[Dict[str, Any]] = [
        {"type": "message_start", "message": {**message, "content": [], "stop_reason": None}}
    ]
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            start: Dict[str, Any] = {"type": "text", "text": ""}
            delta: Dict[str, Any] = {"type": "text_delta", "text": block["text"]}
        else:
            start = {**block, "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}
        events += [
            {"type": "content_block_start", "index": index, "content_block": start},
            {"type": "content_block_delta", "index": index, "delta": delta},
            {"type": "content_block_stop", "index": index},
        ]
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
            "usage": {"output_tokens": 20},
        },
        {"type": "message_stop"},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


class _Server:
    """Answers the Messages API with the given messages, one for each request."""

    def __init__(self, *messages: Dict[str, Any]) -> None:
        self._messages: Iterator[Dict[str, Any]] = iter(messages)
        self.requests: List[httpx2.Request] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(request)
        message = next(self._messages)
        if json.loads(request.content).get("stream"):
            return httpx2.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_server_sent_events(message),
            )
        return httpx2.Response(200, json=message)


def _bad_request(request: httpx2.Request) -> httpx2.Response:
    error = {"type": "invalid_request_error", "message": "bad"}
    return httpx2.Response(400, json={"type": "error", "error": error})


def _client(handler: Any, **kwargs: Any) -> Anthropic:
    transport = httpx2.MockTransport(handler)
    return Anthropic(
        api_key="sk-ant-fake", http_client=httpx2.Client(transport=transport), **kwargs
    )


def _async_client(handler: Any, **kwargs: Any) -> AsyncAnthropic:
    async def async_handler(request: httpx2.Request) -> httpx2.Response:
        await request.aread()
        return handler(request)  # type: ignore[no-any-return]

    transport = httpx2.MockTransport(async_handler)
    return AsyncAnthropic(
        api_key="sk-ant-fake", http_client=httpx2.AsyncClient(transport=transport), **kwargs
    )


def _kwargs(tools: Any, **kwargs: Any) -> Dict[str, Any]:
    return dict(
        model=MODEL,
        max_tokens=1000,
        tools=tools,
        messages=[{"role": "user", "content": "What is the weather?"}],
        **kwargs,
    )


def _attributes(span: ReadableSpan) -> Dict[str, Any]:
    return dict(span.attributes or {})


def _spans(exporter: InMemorySpanExporter, name: str) -> List[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _only(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    (span,) = _spans(exporter, name)
    return span


def _llm_spans(exporter: InMemorySpanExporter) -> List[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if _attributes(span).get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    ]


def _check_chain_span(
    exporter: InMemorySpanExporter,
    *,
    context_attributes: Optional[Mapping[str, Any]] = None,
    with_tools: bool = True,
    extra_input: Optional[Mapping[str, Any]] = None,
) -> ReadableSpan:
    """Checks every attribute of the span of the runner and returns it."""
    span = _only(exporter, RUNNER_SPAN_NAME)
    assert span.status.status_code == StatusCode.OK
    attributes = _attributes(span)
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "CHAIN"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "anthropic"
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == "anthropic"
    # the input has the definitions of the tools, not the objects that define them
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    expected_input = _kwargs([TOOL_DEFINITION] if with_tools else None)
    if not with_tools:
        del expected_input["tools"]
    expected_input.update(extra_input or {})
    assert json.loads(str(attributes.pop(SpanAttributes.INPUT_VALUE))) == expected_input
    # the output is the last message
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    output = json.loads(str(attributes.pop(SpanAttributes.OUTPUT_VALUE)))
    assert output["id"] == "msg_02"
    assert output["content"][0]["text"] == ANSWER
    assert attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert attributes.pop("llm.output_messages.0.message.contents.0.message_content.type") == "text"
    assert attributes.pop("llm.output_messages.0.message.contents.0.message_content.text") == ANSWER
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
    for key, value in (context_attributes or {}).items():
        assert attributes.pop(key) == value
    assert not attributes
    return span


def _check_tool_span(
    exporter: InMemorySpanExporter,
    chain_span: ReadableSpan,
    *,
    masked: bool = False,
    context_attributes: Optional[Mapping[str, Any]] = None,
) -> ReadableSpan:
    """Checks every attribute of the span of the tool and returns it."""
    span = _only(exporter, "get_weather")
    assert span.parent is not None and span.parent.span_id == chain_span.context.span_id
    assert span.status.status_code == StatusCode.OK
    assert not span.events
    attributes = _attributes(span)
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "TOOL"
    assert attributes.pop(SpanAttributes.TOOL_NAME) == "get_weather"
    assert attributes.pop(SpanAttributes.TOOL_DESCRIPTION) == "Get the weather for a location."
    # the definition of the parameters, not the arguments, which are the input
    assert json.loads(str(attributes.pop(SpanAttributes.TOOL_PARAMETERS))) == SCHEMA
    if masked:
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "__REDACTED__"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "__REDACTED__"
    else:
        assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
        mime_type = OpenInferenceMimeTypeValues.TEXT.value
        assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == mime_type
        assert json.loads(str(attributes.pop(SpanAttributes.INPUT_VALUE))) == {"location": SECRET}
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == WEATHER
    for key, value in (context_attributes or {}).items():
        assert attributes.pop(key) == value
    assert not attributes
    return span


def _check_trace(
    exporter: InMemorySpanExporter,
    context_attributes: Optional[Mapping[str, Any]] = None,
) -> ReadableSpan:
    """The runner's span, with the tool's and the two LLM spans as its children, and no others."""
    chain_span = _check_chain_span(exporter, context_attributes=context_attributes)
    _check_tool_span(exporter, chain_span, context_attributes=context_attributes)
    llm_spans = _llm_spans(exporter)
    assert len(llm_spans) == 2
    for span in llm_spans:
        assert span.parent is not None and span.parent.span_id == chain_span.context.span_id
        attributes = _attributes(span)
        for key, value in (context_attributes or {}).items():
            assert attributes[key] == value
    assert len(exporter.get_finished_spans()) == 4
    assert chain_span.parent is None
    return chain_span


CONTEXT_ATTRIBUTES = {
    SpanAttributes.SESSION_ID: "session-id",
    SpanAttributes.USER_ID: "user-id",
    SpanAttributes.METADATA: json.dumps({"key": "value"}),
    SpanAttributes.TAG_TAGS: ("tag",),
}


@pytest.fixture()
def config() -> TraceConfig:
    return TraceConfig()


@pytest.fixture()
def instrumented(
    tracer_provider: TracerProvider, config: TraceConfig
) -> Generator[None, None, None]:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
    yield
    AnthropicInstrumentor().uninstrument()


def _tool_runner(client: Anthropic, tools: Any, **kwargs: Any) -> Any:
    return client.beta.messages.tool_runner(**_kwargs(tools, **kwargs))


def _async_tool_runner(client: AsyncAnthropic, tools: Any, **kwargs: Any) -> Any:
    return client.beta.messages.tool_runner(**_kwargs(tools, **kwargs))


class TestSyncToolRunner:
    def test_until_done(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        message = _tool_runner(client, [get_weather]).until_done()
        assert message.content[0].text == ANSWER
        _check_trace(in_memory_span_exporter)

    def test_for_loop(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        assert len(list(_tool_runner(client, [get_weather]))) == 2
        # the output is recorded on this path too
        _check_trace(in_memory_span_exporter)

    def test_streaming(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        for stream in _tool_runner(client, [get_weather], stream=True):
            for _ in stream:
                pass
        chain_span = _check_chain_span(in_memory_span_exporter, extra_input={"stream": True})
        _check_tool_span(in_memory_span_exporter, chain_span)

    def test_generate_tool_call_response(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        runner = _tool_runner(client, [get_weather])
        for _ in runner:
            runner.generate_tool_call_response()
        _check_trace(in_memory_span_exporter)

    def test_the_span_is_current_only_while_the_runner_runs(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        before = trace_api.get_current_span()
        runner = _tool_runner(client, [get_weather])
        assert trace_api.get_current_span() is before
        next(runner)
        assert trace_api.get_current_span() is before
        runner.until_done()
        assert trace_api.get_current_span() is before
        _check_trace(in_memory_span_exporter)

    def test_an_unused_runner_makes_no_span(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        runner = _tool_runner(_client(_Server()), [get_weather])
        del runner
        gc.collect()
        assert not in_memory_span_exporter.get_finished_spans()

    def test_an_abandoned_runner_ends_its_span(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        server = _Server(_tool_use_message(), _answer_message(), _answer_message())
        client = _client(server)
        runner = _tool_runner(client, [get_weather])
        for _ in runner:
            break
        assert not _spans(in_memory_span_exporter, RUNNER_SPAN_NAME)
        del runner
        gc.collect()
        chain_span = _only(in_memory_span_exporter, RUNNER_SPAN_NAME)
        assert chain_span.end_time is not None
        # a later call is not a child of the span of the runner
        client.beta.messages.create(**{**_kwargs(None), "tools": []})
        assert _llm_spans(in_memory_span_exporter)[-1].parent is None
        assert context_api.get_current() == context_api.Context()

    def test_closing_the_runner_ends_its_span(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        runner = _tool_runner(_client(_Server(_tool_use_message())), [get_weather])
        for _ in runner:
            break
        assert not _spans(in_memory_span_exporter, RUNNER_SPAN_NAME)
        runner.close()
        assert _only(in_memory_span_exporter, RUNNER_SPAN_NAME).end_time is not None

    def test_an_error_in_the_loop_does_not_leak_the_span(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        runner = _tool_runner(_client(_Server(_tool_use_message())), [get_weather])
        with pytest.raises(ValueError):
            for _ in runner:
                raise ValueError("boom")
        del runner
        gc.collect()
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN
        assert _only(in_memory_span_exporter, RUNNER_SPAN_NAME).end_time is not None

    def test_a_failing_tool_records_its_error_once(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message("failing_weather"), _answer_message()))
        _tool_runner(client, [failing_weather]).until_done()
        tool_span = _only(in_memory_span_exporter, "failing_weather")
        assert tool_span.status.status_code == StatusCode.ERROR
        assert [event.name for event in tool_span.events] == ["exception"]
        assert _only(in_memory_span_exporter, RUNNER_SPAN_NAME).status.status_code == StatusCode.OK

    def test_a_failing_request_marks_the_span(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        runner = _tool_runner(_client(_bad_request), [get_weather])
        with pytest.raises(Exception):
            runner.until_done()
        span = _only(in_memory_span_exporter, RUNNER_SPAN_NAME)
        assert span.status.status_code == StatusCode.ERROR
        assert [event.name for event in span.events] == ["exception"]

    def test_one_shot_tools_are_left_to_the_sdk(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        # a one-shot iterable is the SDK's to read, so tracing must not read it first
        client = _client(_Server(_tool_use_message(), _answer_message()))
        runner = _tool_runner(client, (tool for tool in [get_weather]))
        runner.until_done()
        assert not _spans(in_memory_span_exporter, "get_weather")
        chain_span = _check_chain_span(in_memory_span_exporter, with_tools=False)
        assert chain_span.parent is None

    def test_suppressed_tracing(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        token = context_api.attach(
            context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            _tool_runner(client, [get_weather]).until_done()
        finally:
            context_api.detach(token)
        assert not in_memory_span_exporter.get_finished_spans()

    def test_tracing_suppressed_after_the_runner_is_created(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        runner = _tool_runner(client, [get_weather])
        token = context_api.attach(
            context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            runner.until_done()
        finally:
            context_api.detach(token)
        # the tool, which is only traced if tracing is not suppressed when it runs, has no span
        assert not _spans(in_memory_span_exporter, "get_weather")
        assert not _llm_spans(in_memory_span_exporter)

    def test_a_runner_made_before_uninstrument_makes_no_tool_span(
        self, in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
    ) -> None:
        instrumentor = AnthropicInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        client = _client(_Server(_tool_use_message(), _answer_message()))
        runner = _tool_runner(client, [get_weather])
        instrumentor.uninstrument()
        runner.until_done()
        assert not in_memory_span_exporter.get_finished_spans()

    def test_context_attributes(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        with using_attributes(
            session_id="session-id", user_id="user-id", metadata={"key": "value"}, tags=["tag"]
        ):
            runner = _tool_runner(client, [get_weather])
            runner.until_done()
        _check_trace(in_memory_span_exporter, CONTEXT_ATTRIBUTES)

    def test_context_attributes_are_those_of_the_consumer(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        with using_attributes(session_id="session-id", user_id="user-id"):
            runner = _tool_runner(client, [get_weather])
        with using_attributes(
            session_id="session-id", user_id="user-id", metadata={"key": "value"}, tags=["tag"]
        ):
            runner.until_done()
        _check_trace(in_memory_span_exporter, CONTEXT_ATTRIBUTES)

    @pytest.mark.parametrize("config", [TraceConfig(hide_inputs=True, hide_outputs=True)])
    def test_hidden_inputs_and_outputs_do_not_leak(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _client(_Server(_tool_use_message(), _answer_message()))
        _tool_runner(client, [get_weather]).until_done()
        chain_span = _only(in_memory_span_exporter, RUNNER_SPAN_NAME)
        _check_tool_span(in_memory_span_exporter, chain_span, masked=True)
        for span in in_memory_span_exporter.get_finished_spans():
            for key, value in _attributes(span).items():
                assert SECRET not in str(value), (span.name, key)
                assert WEATHER not in str(value), (span.name, key)

    def test_the_sdks_spans_are_not_a_second_set(
        self, in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
    ) -> None:
        pytest.importorskip("anthropic.lib._tracing", reason="no spans of its own before 1.14")
        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
        try:
            client = _client(
                _Server(_tool_use_message(), _answer_message()),
                open_telemetry={"tracer_provider": tracer_provider},
            )
            _tool_runner(client, [get_weather]).until_done()
        finally:
            AnthropicInstrumentor().uninstrument()
        _check_trace(in_memory_span_exporter)

    def test_the_sdks_spans_are_kept_for_a_runner_that_is_not_traced(
        self, in_memory_span_exporter: InMemorySpanExporter, tracer_provider: TracerProvider
    ) -> None:
        pytest.importorskip("anthropic.lib._tracing", reason="no spans of its own before 1.14")
        client = _client(
            _Server(_tool_use_message(), _answer_message()),
            open_telemetry={"tracer_provider": tracer_provider},
        )
        _tool_runner(client, [get_weather]).until_done()
        names = {span.name for span in in_memory_span_exporter.get_finished_spans()}
        assert "anthropic.messages.tool_runner" in names
        assert "anthropic.custom_tool_use get_weather" in names


class TestAsyncToolRunner:
    async def test_until_done(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _async_client(_Server(_tool_use_message(), _answer_message()))
        runner = _async_tool_runner(client, [get_weather_async])
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN
        message = await runner.until_done()
        assert message.content[0].text == ANSWER
        _check_trace(in_memory_span_exporter)

    async def test_for_loop(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _async_client(_Server(_tool_use_message(), _answer_message()))
        runner = _async_tool_runner(client, [get_weather_async])
        assert len([message async for message in runner]) == 2
        _check_trace(in_memory_span_exporter)

    async def test_streaming(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _async_client(_Server(_tool_use_message(), _answer_message()))
        async for stream in _async_tool_runner(client, [get_weather_async], stream=True):
            async for _ in stream:
                pass
        chain_span = _check_chain_span(in_memory_span_exporter, extra_input={"stream": True})
        _check_tool_span(in_memory_span_exporter, chain_span)

    async def test_generate_tool_call_response(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _async_client(_Server(_tool_use_message(), _answer_message()))
        runner = _async_tool_runner(client, [get_weather_async])
        async for _ in runner:
            await runner.generate_tool_call_response()
        # the tool's span is a child of the runner's
        _check_trace(in_memory_span_exporter)

    async def test_consumed_in_another_task(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _async_client(_Server(_tool_use_message(), _answer_message()))
        runner = _async_tool_runner(client, [get_weather_async])
        await asyncio.create_task(runner.until_done())
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN
        _check_trace(in_memory_span_exporter)

    async def test_an_abandoned_runner_ends_its_span(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _async_client(_Server(_tool_use_message(), _answer_message()))
        runner = _async_tool_runner(client, [get_weather_async])
        async for _ in runner:
            break
        del runner
        gc.collect()
        assert _only(in_memory_span_exporter, RUNNER_SPAN_NAME).end_time is not None
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN

    async def test_closing_the_runner_ends_its_span(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        runner = _async_tool_runner(
            _async_client(_Server(_tool_use_message())), [get_weather_async]
        )
        async for _ in runner:
            break
        assert not _spans(in_memory_span_exporter, RUNNER_SPAN_NAME)
        await runner.close()
        assert _only(in_memory_span_exporter, RUNNER_SPAN_NAME).end_time is not None

    async def test_suppressed_tracing(
        self, in_memory_span_exporter: InMemorySpanExporter, instrumented: None
    ) -> None:
        client = _async_client(_Server(_tool_use_message(), _answer_message()))
        runner = _async_tool_runner(client, [get_weather_async])
        token = context_api.attach(
            context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            await runner.until_done()
        finally:
            context_api.detach(token)
        assert not in_memory_span_exporter.get_finished_spans()


def test_the_runners_of_every_client_are_wrapped_and_restored(
    tracer_provider: TracerProvider,
) -> None:
    from anthropic.lib.bedrock import _beta_messages as bedrock
    from anthropic.lib.vertex import _beta_messages as vertex
    from anthropic.resources.beta import messages

    classes = [
        messages.Messages,
        messages.AsyncMessages,
        bedrock.Messages,
        bedrock.AsyncMessages,
        vertex.Messages,
        vertex.AsyncMessages,
    ]
    originals = [cls.__dict__["tool_runner"] for cls in classes]
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    try:
        assert all(
            cls.__dict__["tool_runner"] is not original for cls, original in zip(classes, originals)
        )
    finally:
        instrumentor.uninstrument()
    assert [cls.__dict__["tool_runner"] for cls in classes] == originals
