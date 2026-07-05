import json
from typing import Any, Iterator

import pytest
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
from together import AsyncTogether, Together
from together.types import ChatCompletionResponse
from together.types.chat.chat_completion import Choice, ChoiceMessage
from together.types.chat.chat_completion_usage import ChatCompletionUsage
from together.types.tool_choice import Function, ToolChoice

from openinference.instrumentation import OITracer
from openinference.instrumentation.together import TogetherInstrumentor


def _text_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="c-0",
        object="chat.completion",
        created=0,
        prompt=[],
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChoiceMessage(
                    role="assistant", content="The sky is blue because of Rayleigh scattering."
                ),
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=26, completion_tokens=12, total_tokens=38),
    )


def _tool_response() -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="c-1",
        object="chat.completion",
        created=0,
        prompt=[],
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChoiceMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolChoice(
                            id="call-1",
                            index=0,
                            type="function",
                            function=Function(
                                name="get_current_weather",
                                arguments=json.dumps({"city": "Paris"}),
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=ChatCompletionUsage(prompt_tokens=40, completion_tokens=8, total_tokens=48),
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
    TogetherInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    TogetherInstrumentor().uninstrument()


def test_oitracer() -> None:
    assert isinstance(TogetherInstrumentor()._tracer, OITracer)


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="together")
    assert isinstance(entrypoint.load()(), TogetherInstrumentor)


def test_chat(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = Together(api_key="fake-key")
    monkeypatch.setattr(client.chat.completions, "_post", lambda *a, **k: _text_response())
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )
    assert response.choices[0].message.content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert spans[0].name == "chat"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert "Llama-3.3-70B" in attrs[SpanAttributes.LLM_MODEL_NAME]
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
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
    client = Together(api_key="fake-key")
    monkeypatch.setattr(client.chat.completions, "_post", lambda *a, **k: _tool_response())
    client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[{"type": "function", "function": {"name": "get_current_weather"}}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}"] == "call-1"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_current_weather"
    assert json.loads(
        attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
    ) == {"city": "Paris"}
    assert attrs["llm.tools.0.tool.json_schema"]


async def test_async_chat(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _mock_post(*a: Any, **k: Any) -> ChatCompletionResponse:
        return _text_response()

    client = AsyncTogether(api_key="fake-key")
    monkeypatch.setattr(client.chat.completions, "_post", _mock_post)
    await client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "async_chat"
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38


def test_suppress_tracing(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from openinference.instrumentation import suppress_tracing

    client = Together(api_key="fake-key")
    monkeypatch.setattr(client.chat.completions, "_post", lambda *a, **k: _text_response())
    with suppress_tracing():
        client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
        )
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
