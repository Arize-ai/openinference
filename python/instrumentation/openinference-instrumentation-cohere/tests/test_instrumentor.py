import json
from types import SimpleNamespace
from typing import Any, Iterator

import cohere
import pytest
from cohere.types import (
    AssistantMessageResponse,
    TextAssistantMessageResponseContentItem,
    ToolCallV2,
    ToolCallV2Function,
    Usage,
    UsageTokens,
)
from cohere.v2.raw_client import AsyncRawV2Client, RawV2Client
from cohere.v2.types import V2ChatResponse
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer
from openinference.instrumentation.cohere import CohereInstrumentor


def _text_response() -> V2ChatResponse:
    return V2ChatResponse(
        id="c-0",
        finish_reason="COMPLETE",
        message=AssistantMessageResponse(
            role="assistant",
            content=[
                TextAssistantMessageResponseContentItem(
                    type="text", text="The sky is blue because of Rayleigh scattering."
                )
            ],
        ),
        usage=Usage(tokens=UsageTokens(input_tokens=26, output_tokens=12)),
    )


def _tool_response() -> V2ChatResponse:
    return V2ChatResponse(
        id="c-1",
        finish_reason="TOOL_CALL",
        message=AssistantMessageResponse(
            role="assistant",
            content=[],
            tool_calls=[
                ToolCallV2(
                    id="call-1",
                    type="function",
                    function=ToolCallV2Function(
                        name="get_current_weather",
                        arguments=json.dumps({"city": "Paris"}),
                    ),
                )
            ],
        ),
        usage=Usage(tokens=UsageTokens(input_tokens=40, output_tokens=8)),
    )


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(tracer_provider: TracerProvider) -> Iterator[None]:
    CohereInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    CohereInstrumentor().uninstrument()


def _client() -> "cohere.ClientV2":
    return cohere.ClientV2(api_key="fake-key")


def test_oitracer() -> None:
    assert isinstance(CohereInstrumentor()._tracer, OITracer)


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="cohere")
    assert isinstance(entrypoint.load()(), CohereInstrumentor)


def test_chat(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )

    response = _client().chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )
    assert response.message.content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert spans[0].name == "chat"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "command-r-plus"
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    )
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "Why is the sky blue?"
    )
    assert (
        "Rayleigh"
        in attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
    )
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 26
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 12
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38
    assert SpanAttributes.INPUT_VALUE in attrs
    assert SpanAttributes.OUTPUT_VALUE in attrs


def test_chat_with_tool_call(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_tool_response())
    )

    _client().chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[{"type": "function", "function": {"name": "get_current_weather"}}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call-1"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_current_weather"
    raw_args = attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
    assert json.loads(raw_args) == {"city": "Paris"}
    assert attrs["llm.tools.0.tool.json_schema"]


async def test_async_chat(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _mock_chat(self: Any, **k: Any) -> Any:
        return SimpleNamespace(data=_text_response())

    monkeypatch.setattr(AsyncRawV2Client, "chat", _mock_chat)

    await cohere.AsyncClientV2(api_key="fake-key").chat(
        model="command-r-plus",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "async_chat"
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "command-r-plus"
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38


def test_suppress_tracing(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openinference.instrumentation import suppress_tracing

    monkeypatch.setattr(
        RawV2Client, "chat", lambda self, **k: SimpleNamespace(data=_text_response())
    )
    with suppress_tracing():
        _client().chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
        )
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
