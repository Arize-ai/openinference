import json
from typing import Any, Iterator

import ollama
import pytest
from ollama import ChatResponse, Message
from ollama._client import AsyncClient, Client
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
from openinference.instrumentation.ollama import OllamaInstrumentor


def _text_response() -> ChatResponse:
    return ChatResponse(
        model="llama3.2",
        message=Message(
            role="assistant", content="The sky is blue because of Rayleigh scattering."
        ),
        done=True,
        done_reason="stop",
        prompt_eval_count=26,
        eval_count=12,
    )


def _tool_response() -> ChatResponse:
    return ChatResponse(
        model="llama3.2",
        message=Message(
            role="assistant",
            content="",
            tool_calls=[
                Message.ToolCall(
                    function=Message.ToolCall.Function(
                        name="get_current_weather",
                        arguments={"city": "Paris"},
                    )
                )
            ],
        ),
        done=True,
        done_reason="stop",
        prompt_eval_count=40,
        eval_count=8,
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
    OllamaInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OllamaInstrumentor().uninstrument()


def test_oitracer(tracer_provider: TracerProvider) -> None:
    assert isinstance(OllamaInstrumentor()._tracer, OITracer)


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (instrumentor_entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="ollama")
    instrumentor = instrumentor_entrypoint.load()()
    assert isinstance(instrumentor, OllamaInstrumentor)


def test_chat(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Client, "_request", lambda self, *a, **k: _text_response())

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )
    assert response.message.content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert spans[0].name == "chat"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "llama3.2"
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    )
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "Why is the sky blue?"
    )
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"]
        == "assistant"
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
    monkeypatch.setattr(Client, "_request", lambda self, *a, **k: _tool_response())

    ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[{"type": "function", "function": {"name": "get_current_weather"}}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_current_weather"
    raw_args = attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
    assert json.loads(raw_args) == {"city": "Paris"}
    # The tool schema is recorded on the request side.
    assert attrs["llm.tools.0.tool.json_schema"]


async def test_async_chat(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _mock_request(self: Any, *a: Any, **k: Any) -> ChatResponse:
        return _text_response()

    monkeypatch.setattr(AsyncClient, "_request", _mock_request)

    await ollama.AsyncClient().chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "async_chat"
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "llama3.2"
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38


def test_suppress_tracing(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openinference.instrumentation import suppress_tracing

    monkeypatch.setattr(Client, "_request", lambda self, *a, **k: _text_response())
    with suppress_tracing():
        ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
        )
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
