import gc
import json
from typing import Any, Iterator

import ollama
import pytest
from ollama import ChatResponse, Message
from ollama._client import AsyncClient, Client
from opentelemetry import trace as trace_api
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
    OpenInferenceLLMProviderValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

# The committed cassettes in tests/cassettes/ were recorded against a local
# Ollama server running this model. To re-record, start `ollama serve`, pull
# the model, and run pytest with `--record-mode=rewrite`.
MODEL = "qwen3.6:latest"
OPTIONS = {"temperature": 0, "seed": 42}


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


@pytest.mark.vcr
def test_chat(in_memory_span_exporter: InMemorySpanExporter) -> None:
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}],
        options=OPTIONS,
    )
    assert response.message.content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    )
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "Why is the sky blue? Answer in one sentence."
    )
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"]
        == "assistant"
    )
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == response.message.content
    )
    prompt_tokens = attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT]
    completion_tokens = attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION]
    assert isinstance(prompt_tokens, int) and prompt_tokens > 0
    assert isinstance(completion_tokens, int) and completion_tokens > 0
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == prompt_tokens + completion_tokens
    assert SpanAttributes.INPUT_VALUE in attrs
    assert SpanAttributes.OUTPUT_VALUE in attrs


@pytest.mark.vcr
def test_chat_with_tool_call(in_memory_span_exporter: InMemorySpanExporter) -> None:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": "What is the weather in Paris? Use the get_current_weather tool.",
            }
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the current weather in a city.",
                    "parameters": {
                        "type": "object",
                        "required": ["city"],
                        "properties": {"city": {"type": "string", "description": "The city name."}},
                    },
                },
            }
        ],
        options=OPTIONS,
    )
    assert response.message.tool_calls

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_current_weather"
    raw_args = attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]
    assert isinstance(json.loads(str(raw_args)), dict)
    # The tool schema is recorded on the request side.
    schema = json.loads(
        str(attrs[f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"])
    )
    assert schema["function"]["name"] == "get_current_weather"


@pytest.mark.vcr
def test_chat_with_callable_tool(in_memory_span_exporter: InMemorySpanExporter) -> None:
    def add_two_numbers(a: int, b: int) -> int:
        """Add two numbers.

        Args:
            a: The first number.
            b: The second number.
        """
        return a + b

    ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "What is 2 + 3? Use the add_two_numbers tool."}],
        tools=[add_two_numbers],
        options=OPTIONS,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    schema = json.loads(
        str(attrs[f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"])
    )
    # The schema must be a JSON tool definition, not the function's repr.
    assert schema["function"]["name"] == "add_two_numbers"
    assert schema["function"]["parameters"]["required"] == ["a", "b"]


@pytest.mark.vcr
async def test_async_chat(in_memory_span_exporter: InMemorySpanExporter) -> None:
    response = await AsyncClient().chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}],
        options=OPTIONS,
    )
    assert response.message.content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "AsyncChat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in attrs


@pytest.mark.vcr
def test_chat_stream(in_memory_span_exporter: InMemorySpanExporter) -> None:
    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5, digits only."}],
        stream=True,
        options=OPTIONS,
    )
    # The span must not finish until the stream is exhausted.
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
    chunks = list(stream)
    assert len(chunks) > 1

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    streamed_content = "".join(chunk.message.content or "" for chunk in chunks)
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == streamed_content
    )
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in attrs
    assert "generator" not in str(attrs[SpanAttributes.OUTPUT_VALUE])


@pytest.mark.vcr
async def test_async_chat_stream(in_memory_span_exporter: InMemorySpanExporter) -> None:
    stream = await AsyncClient().chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Count from 1 to 5, digits only."}],
        stream=True,
        options=OPTIONS,
    )
    chunks = [chunk async for chunk in stream]
    assert len(chunks) > 1

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "AsyncChat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    streamed_content = "".join(chunk.message.content or "" for chunk in chunks)
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == streamed_content
    )
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL in attrs


@pytest.mark.vcr
def test_chat_error_records_model_name(in_memory_span_exporter: InMemorySpanExporter) -> None:
    with pytest.raises(Exception):
        ollama.chat(
            model="nonexistent-model:latest",
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == "nonexistent-model:latest"


@pytest.mark.vcr
def test_suppress_tracing(in_memory_span_exporter: InMemorySpanExporter) -> None:
    from openinference.instrumentation import suppress_tracing

    with suppress_tracing():
        ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}],
            options=OPTIONS,
        )
    assert len(in_memory_span_exporter.get_finished_spans()) == 0


@pytest.mark.vcr
def test_context_attributes_propagation(in_memory_span_exporter: InMemorySpanExporter) -> None:
    with using_attributes(
        session_id="my-session",
        user_id="my-user",
        metadata={"env": "test"},
        tags=["tag-1", "tag-2"],
    ):
        ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}],
            options=OPTIONS,
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attrs[SpanAttributes.SESSION_ID] == "my-session"
    assert attrs[SpanAttributes.USER_ID] == "my-user"
    assert json.loads(str(attrs[SpanAttributes.METADATA])) == {"env": "test"}
    assert list(attrs[SpanAttributes.TAG_TAGS]) == ["tag-1", "tag-2"]  # type: ignore[arg-type]


@pytest.mark.vcr
def test_trace_config_masking(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
) -> None:
    instrumentor = OllamaInstrumentor()
    instrumentor.uninstrument()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )
    ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "This contains PII"}],
        options=OPTIONS,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}" not in attrs
    assert (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}" not in attrs
    )


# ---------------------------------------------------------------------------
# Fault-injection tests below monkeypatch the client transport because the
# scenarios (abandoned streams, mid-stream failures, thinking-model output)
# cannot be captured deterministically in a vcr cassette.
# ---------------------------------------------------------------------------


def test_chat_stream_abandoned_before_iteration(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stream_chunks(self: Any, *a: Any, **k: Any) -> Iterator[ChatResponse]:
        yield ChatResponse(
            model=MODEL,
            message=Message(role="assistant", content="hi"),
            done=True,
        )

    monkeypatch.setattr(Client, "_request", _stream_chunks)

    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "Why is the sky blue?"}],
        stream=True,
    )
    # Drop the stream without ever iterating it: the span must still be
    # finished (with status UNSET, to distinguish it from a completed stream).
    del stream
    gc.collect()

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.UNSET
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL


def test_chat_stream_error_keeps_partial_output(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stream_chunks(self: Any, *a: Any, **k: Any) -> Iterator[ChatResponse]:
        yield ChatResponse(
            model=MODEL,
            message=Message(role="assistant", content="The sky "),
            done=False,
        )
        raise RuntimeError("connection dropped")

    monkeypatch.setattr(Client, "_request", _stream_chunks)

    with pytest.raises(RuntimeError):
        list(
            ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": "Why is the sky blue?"}],
                stream=True,
            )
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    # The partial output that arrived before the failure is preserved.
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == "The sky "
    )
    assert SpanAttributes.OUTPUT_VALUE in attrs


def test_chat_stream_merges_thinking(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _stream_chunks(self: Any, *a: Any, **k: Any) -> Iterator[ChatResponse]:
        yield ChatResponse(
            model=MODEL,
            message=Message(role="assistant", content="", thinking="Let me "),
            done=False,
        )
        yield ChatResponse(
            model=MODEL,
            message=Message(role="assistant", content="Blue.", thinking="think."),
            done=True,
            done_reason="stop",
            prompt_eval_count=26,
            eval_count=12,
        )

    monkeypatch.setattr(Client, "_request", _stream_chunks)

    list(
        ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
            stream=True,
        )
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Chat"
    assert span.status.status_code == trace_api.StatusCode.OK
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.OLLAMA.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    output = json.loads(str(attrs[SpanAttributes.OUTPUT_VALUE]))
    assert output["message"]["thinking"] == "Let me think."
    assert output["message"]["content"] == "Blue."


@pytest.mark.parametrize("done_reason", ["stop", "length"])
def test_finish_reason_values(
    done_reason: str,
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Client,
        "_request",
        lambda self, *a, **k: ChatResponse(
            model=MODEL,
            message=Message(role="assistant", content="hi"),
            done=True,
            done_reason=done_reason,
            prompt_eval_count=5,
            eval_count=2,
        ),
    )

    ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs.get(SpanAttributes.LLM_FINISH_REASON) == done_reason


def test_uninstrument_deactivates_captured_aliases(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Client,
        "_request",
        lambda self, *a, **k: ChatResponse(
            model=MODEL, message=Message(role="assistant", content="hi"), done=True
        ),
    )
    # Capture an alias WHILE instrumented: it points at the wrapper.
    captured_chat = ollama.chat
    OllamaInstrumentor().uninstrument()
    try:
        captured_chat(
            model=MODEL,
            messages=[{"role": "user", "content": "Why is the sky blue?"}],
        )
        # The deactivated wrapper must pass through without tracing.
        assert len(in_memory_span_exporter.get_finished_spans()) == 0
    finally:
        # Restore for the autouse fixture's teardown symmetry.
        OllamaInstrumentor().instrument()
