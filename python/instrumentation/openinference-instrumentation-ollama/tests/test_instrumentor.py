import json
from typing import Any, Iterator

import ollama
import pytest
from ollama import ChatResponse, Message
from ollama._client import AsyncClient, Client
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import (
    REDACTED_VALUE,
    OITracer,
    TraceConfig,
    using_attributes,
)
from openinference.instrumentation.ollama import OllamaInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)


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
    assert attrs[SpanAttributes.LLM_PROVIDER] == "ollama"
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
    assert "Rayleigh" in str(
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
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
    assert json.loads(str(raw_args)) == {"city": "Paris"}
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


def test_chat_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stream_chunks(self: Any, *a: Any, **k: Any) -> Iterator[ChatResponse]:
        yield ChatResponse(
            model="llama3.2",
            message=Message(role="assistant", content="The sky "),
            done=False,
        )
        yield ChatResponse(
            model="llama3.2",
            message=Message(role="assistant", content="is blue."),
            done=True,
            done_reason="stop",
            prompt_eval_count=26,
            eval_count=12,
        )

    monkeypatch.setattr(Client, "_request", _stream_chunks)

    stream = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        stream=True,
    )
    # The span must not finish until the stream is exhausted.
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
    chunks = list(stream)
    assert len(chunks) == 2

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "llama3.2"
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "The sky is blue."
    )
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 26
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 12
    assert "generator" not in str(attrs[SpanAttributes.OUTPUT_VALUE])


async def test_async_chat_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _stream_chunks(*a: Any, **k: Any) -> Any:
        async def _inner() -> Any:
            yield ChatResponse(
                model="llama3.2",
                message=Message(role="assistant", content="The sky "),
                done=False,
            )
            yield ChatResponse(
                model="llama3.2",
                message=Message(role="assistant", content="is blue."),
                done=True,
                done_reason="stop",
                prompt_eval_count=26,
                eval_count=12,
            )

        return _inner()

    monkeypatch.setattr(AsyncClient, "_request", _stream_chunks)

    stream = await ollama.AsyncClient().chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        stream=True,
    )
    chunks = [chunk async for chunk in stream]
    assert len(chunks) == 2

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "The sky is blue."
    )
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38


def test_chat_error_records_model_name(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(self: Any, *a: Any, **k: Any) -> ChatResponse:
        raise RuntimeError("boom")

    monkeypatch.setattr(Client, "_request", _raise)
    with pytest.raises(RuntimeError):
        ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert not spans[0].status.is_ok
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "llama3.2"


def test_chat_with_callable_tool(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get_current_weather(city: str) -> str:
        """Get the current weather in a city.

        Args:
            city: The name of the city.
        """
        return "sunny"

    monkeypatch.setattr(Client, "_request", lambda self, *a, **k: _tool_response())
    ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[get_current_weather],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    schema = json.loads(str(attrs["llm.tools.0.tool.json_schema"]))
    # The schema must be a JSON tool definition, not the function's repr.
    assert isinstance(schema, dict)
    assert "get_current_weather" in json.dumps(schema)


def test_context_attributes_propagation(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Client, "_request", lambda self, *a, **k: _text_response())
    with using_attributes(
        session_id="my-session",
        user_id="my-user",
        metadata={"env": "test"},
        tags=["tag-1", "tag-2"],
    ):
        ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.SESSION_ID] == "my-session"
    assert attrs[SpanAttributes.USER_ID] == "my-user"
    assert json.loads(str(attrs[SpanAttributes.METADATA])) == {"env": "test"}
    assert list(attrs[SpanAttributes.TAG_TAGS]) == ["tag-1", "tag-2"]  # type: ignore[arg-type]


def test_trace_config_masking(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumentor = OllamaInstrumentor()
    instrumentor.uninstrument()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )
    monkeypatch.setattr(Client, "_request", lambda self, *a, **k: _text_response())
    ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": "This contains PII"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}" not in attrs
    assert (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}" not in attrs
    )
