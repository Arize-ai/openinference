import gc
import json
from typing import Any, Dict, Generator, List, Mapping, Optional

import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.lib.tools import beta_async_tool, beta_tool
from httpx import Response
from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from respx import MockRouter

from openinference.instrumentation import TraceConfig, using_attributes
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)

MODEL = "claude-3-5-sonnet-20241022"
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
LLM_SPAN_NAMES = ("beta.messages.parse", "beta.messages.create")


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


def _message(message_id: str, content: List[Dict[str, Any]], stop_reason: str) -> Response:
    return Response(
        200,
        json={
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": MODEL,
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
    )


def _tool_use_response(name: str = "get_weather", location: str = SECRET) -> Response:
    tool_use = {"type": "tool_use", "id": "toolu_01", "name": name, "input": {"location": location}}
    return _message("msg_01", [tool_use], "tool_use")


def _answer_response() -> Response:
    return _message("msg_02", [{"type": "text", "text": ANSWER}], "end_turn")


def _mock_messages(respx_mock: MockRouter, *responses: Response) -> None:
    respx_mock.post("https://api.anthropic.com/v1/messages").side_effect = list(responses)


def _runner_kwargs(tools: Any) -> Dict[str, Any]:
    return dict(
        model=MODEL,
        max_tokens=1000,
        tools=tools,
        messages=[{"role": "user", "content": "What is the weather?"}],
    )


def _attributes(span: ReadableSpan) -> Dict[str, Any]:
    return dict(span.attributes or {})


def _spans(exporter: InMemorySpanExporter, name: str) -> List[ReadableSpan]:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def _only(exporter: InMemorySpanExporter, name: str) -> ReadableSpan:
    (span,) = _spans(exporter, name)
    return span


def _children(exporter: InMemorySpanExporter, parent: ReadableSpan) -> List[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if span.parent and span.parent.span_id == parent.context.span_id
    ]


def _check_chain_span(
    exporter: InMemorySpanExporter,
    *,
    expected_context_attributes: Optional[Mapping[str, Any]] = None,
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
    assert json.loads(str(attributes.pop(SpanAttributes.INPUT_VALUE))) == {
        **_runner_kwargs([TOOL_DEFINITION]),
    }
    # the output is the last message
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    output = json.loads(str(attributes.pop(SpanAttributes.OUTPUT_VALUE)))
    assert output["id"] == "msg_02"
    assert output["content"][0]["text"] == ANSWER
    assert attributes.pop("llm.output_messages.0.message.role") == "assistant"
    assert attributes.pop("llm.output_messages.0.message.contents.0.message_content.type") == "text"
    assert attributes.pop("llm.output_messages.0.message.contents.0.message_content.text") == ANSWER
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
    for key, value in (expected_context_attributes or {}).items():
        assert attributes.pop(key) == value
    assert not attributes
    return span


def _check_tool_span(
    exporter: InMemorySpanExporter,
    chain_span: ReadableSpan,
    *,
    masked: bool = False,
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
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.TEXT.value
    if masked:
        assert attributes.pop(SpanAttributes.INPUT_VALUE) == "__REDACTED__"
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "__REDACTED__"
    else:
        assert json.loads(str(attributes.pop(SpanAttributes.INPUT_VALUE))) == {"location": SECRET}
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == WEATHER
    assert not attributes
    return span


def _check_llm_spans(exporter: InMemorySpanExporter, chain_span: ReadableSpan) -> None:
    llm_spans = [span for span in _children(exporter, chain_span) if span.name in LLM_SPAN_NAMES]
    assert len(llm_spans) == 2
    for span in llm_spans:
        assert _attributes(span)[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "LLM"


def _check_trace(exporter: InMemorySpanExporter) -> ReadableSpan:
    chain_span = _check_chain_span(exporter)
    _check_tool_span(exporter, chain_span)
    _check_llm_spans(exporter, chain_span)
    assert len(exporter.get_finished_spans()) == 4
    assert chain_span.parent is None
    return chain_span


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


class TestSyncToolRunner:
    def test_until_done(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = Anthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([get_weather])
        )
        message = runner.until_done()
        assert message.content[0].text == ANSWER  # type: ignore[union-attr]
        _check_trace(in_memory_span_exporter)

    def test_for_loop(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = Anthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([get_weather])
        )
        assert len(list(runner)) == 2
        # the output is recorded on this path too
        _check_trace(in_memory_span_exporter)

    def test_one_shot_tools_are_recorded(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = Anthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs(tool for tool in [get_weather])
        )
        runner.until_done()
        _check_trace(in_memory_span_exporter)

    def test_the_span_is_current_only_while_the_runner_runs(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        client = Anthropic(api_key="sk-ant-fake")
        before = trace_api.get_current_span()
        runner = client.beta.messages.tool_runner(**_runner_kwargs([get_weather]))
        assert trace_api.get_current_span() is before
        next(runner)
        assert trace_api.get_current_span() is before
        runner.until_done()
        assert trace_api.get_current_span() is before
        # a span of the caller, started before the runner runs, is not a child of the runner
        _check_trace(in_memory_span_exporter)

    def test_an_abandoned_runner_ends_its_span(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        client = Anthropic(api_key="sk-ant-fake")
        runner = client.beta.messages.tool_runner(**_runner_kwargs([get_weather]))
        for _ in runner:
            break
        assert not _spans(in_memory_span_exporter, RUNNER_SPAN_NAME)
        del runner
        gc.collect()
        chain_span = _only(in_memory_span_exporter, RUNNER_SPAN_NAME)
        assert chain_span.end_time is not None
        # a later call is not a child of the span of the runner
        _mock_messages(respx_mock, _answer_response())
        client.beta.messages.create(**{**_runner_kwargs([]), "tools": []})
        later = _spans(in_memory_span_exporter, "beta.messages.create")[-1]
        assert later.parent is None
        assert context_api.get_current() == context_api.Context()

    def test_an_error_in_the_loop_does_not_leak_the_span(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = Anthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([get_weather])
        )
        with pytest.raises(ValueError):
            for _ in runner:
                raise ValueError("boom")
        del runner
        gc.collect()
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN
        assert _only(in_memory_span_exporter, RUNNER_SPAN_NAME).end_time is not None

    def test_a_failing_tool_records_its_error_once(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response("failing_weather"), _answer_response())
        runner = Anthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([failing_weather])
        )
        runner.until_done()
        tool_span = _only(in_memory_span_exporter, "failing_weather")
        assert tool_span.status.status_code == StatusCode.ERROR
        assert [event.name for event in tool_span.events] == ["exception"]
        assert _only(in_memory_span_exporter, RUNNER_SPAN_NAME).status.status_code == StatusCode.OK

    def test_a_failing_request_marks_the_span(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        respx_mock.post("https://api.anthropic.com/v1/messages").side_effect = [
            Response(400, json={"type": "error", "error": {"type": "x", "message": "bad"}})
        ]
        runner = Anthropic(api_key="sk-ant-fake", max_retries=0).beta.messages.tool_runner(
            **_runner_kwargs([get_weather])
        )
        with pytest.raises(Exception):
            runner.until_done()
        span = _only(in_memory_span_exporter, RUNNER_SPAN_NAME)
        assert span.status.status_code == StatusCode.ERROR
        assert [event.name for event in span.events] == ["exception"]

    def test_suppressed_tracing(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        token = context_api.attach(
            context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            runner = Anthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
                **_runner_kwargs([get_weather])
            )
            runner.until_done()
        finally:
            context_api.detach(token)
        assert not in_memory_span_exporter.get_finished_spans()

    def test_context_attributes(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        with using_attributes(
            session_id="session-id",
            user_id="user-id",
            metadata={"key": "value"},
            tags=["tag"],
        ):
            runner = Anthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
                **_runner_kwargs([get_weather])
            )
            runner.until_done()
        chain_span = _check_chain_span(
            in_memory_span_exporter,
            expected_context_attributes={
                SpanAttributes.SESSION_ID: "session-id",
                SpanAttributes.USER_ID: "user-id",
                SpanAttributes.METADATA: json.dumps({"key": "value"}),
                SpanAttributes.TAG_TAGS: ("tag",),
            },
        )
        assert _attributes(chain_span)[SpanAttributes.SESSION_ID] == "session-id"

    @pytest.mark.parametrize("config", [TraceConfig(hide_inputs=True, hide_outputs=True)])
    def test_hidden_inputs_and_outputs_do_not_leak(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = Anthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([get_weather])
        )
        runner.until_done()
        tool_span = _only(in_memory_span_exporter, "get_weather")
        attributes = _attributes(tool_span)
        assert attributes[SpanAttributes.INPUT_VALUE] == "__REDACTED__"
        assert attributes[SpanAttributes.OUTPUT_VALUE] == "__REDACTED__"
        for span in in_memory_span_exporter.get_finished_spans():
            for key, value in _attributes(span).items():
                assert SECRET not in str(value), (span.name, key)
                assert WEATHER not in str(value), (span.name, key)
        # the definition of the parameters is not hidden, the arguments are not in it
        assert json.loads(str(attributes[SpanAttributes.TOOL_PARAMETERS])) == SCHEMA


class TestAsyncToolRunner:
    async def test_until_done(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = AsyncAnthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([get_weather_async])
        )
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN
        message = await runner.until_done()
        assert message.content[0].text == ANSWER  # type: ignore[union-attr]
        _check_trace(in_memory_span_exporter)

    async def test_for_loop(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = AsyncAnthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([get_weather_async])
        )
        assert len([message async for message in runner]) == 2
        _check_trace(in_memory_span_exporter)

    async def test_consumed_in_another_task(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        import asyncio

        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = AsyncAnthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([get_weather_async])
        )
        await asyncio.create_task(runner.until_done())
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN
        _check_trace(in_memory_span_exporter)

    async def test_an_abandoned_runner_ends_its_span(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        runner = AsyncAnthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
            **_runner_kwargs([get_weather_async])
        )
        async for _ in runner:
            break
        del runner
        gc.collect()
        assert _only(in_memory_span_exporter, RUNNER_SPAN_NAME).end_time is not None
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN

    async def test_suppressed_tracing(
        self,
        respx_mock: MockRouter,
        in_memory_span_exporter: InMemorySpanExporter,
        instrumented: None,
    ) -> None:
        _mock_messages(respx_mock, _tool_use_response(), _answer_response())
        token = context_api.attach(
            context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
        )
        try:
            runner = AsyncAnthropic(api_key="sk-ant-fake").beta.messages.tool_runner(
                **_runner_kwargs([get_weather_async])
            )
            await runner.until_done()
        finally:
            context_api.detach(token)
        assert not in_memory_span_exporter.get_finished_spans()


def test_uninstrument_restores_the_tool_runner(tracer_provider: TracerProvider) -> None:
    from anthropic.resources.beta.messages import AsyncMessages, Messages

    original = (Messages.tool_runner, AsyncMessages.tool_runner)
    instrumentor = AnthropicInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    assert Messages.tool_runner is not original[0]
    instrumentor.uninstrument()
    assert (Messages.tool_runner, AsyncMessages.tool_runner) == original
