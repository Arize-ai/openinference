import json
from typing import List

import pytest
import respx
from httpx import Response
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from together import AsyncTogether, Together
from together.types import ToolsParam

from openinference.instrumentation import OITracer, TraceConfig, using_attributes
from openinference.instrumentation.config import REDACTED_VALUE
from openinference.instrumentation.together import TogetherInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)

_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

_TOOLS: List[ToolsParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city, e.g. Paris"},
                },
                "required": ["city"],
            },
        },
    }
]


def test_oitracer() -> None:
    assert isinstance(TogetherInstrumentor()._tracer, OITracer)


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="together")
    assert isinstance(entrypoint.load()(), TogetherInstrumentor)


@pytest.mark.vcr
def test_chat(in_memory_span_exporter: InMemorySpanExporter) -> None:
    client = Together()
    response = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}],
    )
    assert response.choices[0].message is not None
    assert response.choices[0].message.content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Completions"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.TOGETHER.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == _MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert (
        attrs[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "user"
    )
    assert attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
    assert isinstance(attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT], int)
    assert isinstance(attrs[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION], int)
    assert isinstance(attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL], int)
    assert SpanAttributes.INPUT_VALUE in attrs
    assert SpanAttributes.OUTPUT_VALUE in attrs
    invocation_parameters = json.loads(str(attrs[SpanAttributes.LLM_INVOCATION_PARAMETERS]))
    assert invocation_parameters["model"] == _MODEL
    # Unset parameters (Omit/NotGiven sentinels) must not leak into the span.
    assert "Omit" not in str(attrs[SpanAttributes.INPUT_VALUE])
    assert "NotGiven" not in str(attrs[SpanAttributes.INPUT_VALUE])


@pytest.mark.vcr
def test_chat_with_tool_call(in_memory_span_exporter: InMemorySpanExporter) -> None:
    client = Together()
    client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": "What is the weather in Paris right now?"}],
        tools=_TOOLS,
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Completions"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.TOGETHER.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == _MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "tool_calls"
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}"]
    assert attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] == "get_current_weather"
    arguments = json.loads(
        str(attrs[f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"])
    )
    assert arguments["city"]
    assert attrs["llm.tools.0.tool.json_schema"]


@pytest.mark.vcr
async def test_async_chat(in_memory_span_exporter: InMemorySpanExporter) -> None:
    client = AsyncTogether()
    response = await client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}],
    )
    assert response.choices[0].message is not None
    assert response.choices[0].message.content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "AsyncCompletions"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.TOGETHER.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == _MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert isinstance(attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL], int)


@pytest.mark.vcr
def test_chat_stream(in_memory_span_exporter: InMemorySpanExporter) -> None:
    client = Together()
    stream = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": "Count from one to five."}],
        stream=True,
    )
    # No span should be exported before the stream is consumed.
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
    content = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    assert content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Completions"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.TOGETHER.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == _MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attrs[SpanAttributes.OUTPUT_VALUE] == content
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"]
        == "assistant"
    )
    assert (
        attrs[f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"]
        == content
    )


@pytest.mark.vcr
async def test_async_chat_stream(in_memory_span_exporter: InMemorySpanExporter) -> None:
    client = AsyncTogether()
    stream = await client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": "Count from one to five."}],
        stream=True,
    )
    assert len(in_memory_span_exporter.get_finished_spans()) == 0
    content = ""
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content += chunk.choices[0].delta.content
    assert content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "AsyncCompletions"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.TOGETHER.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == _MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attrs[SpanAttributes.OUTPUT_VALUE] == content


@pytest.mark.vcr
def test_suppress_tracing(in_memory_span_exporter: InMemorySpanExporter) -> None:
    from openinference.instrumentation import suppress_tracing

    client = Together()
    with suppress_tracing():
        client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}],
        )
    assert len(in_memory_span_exporter.get_finished_spans()) == 0


@pytest.mark.vcr
def test_context_attributes_propagation(in_memory_span_exporter: InMemorySpanExporter) -> None:
    client = Together()
    with using_attributes(
        session_id="session-1",
        user_id="user-1",
        metadata={"env": "test"},
        tags=["tag-1", "tag-2"],
    ):
        client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "Why is the sky blue? Answer in one sentence."}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Completions"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.TOGETHER.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == _MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attrs[SpanAttributes.SESSION_ID] == "session-1"
    assert attrs[SpanAttributes.USER_ID] == "user-1"
    assert json.loads(str(attrs[SpanAttributes.METADATA])) == {"env": "test"}
    assert list(attrs[SpanAttributes.TAG_TAGS]) == ["tag-1", "tag-2"]  # type: ignore[arg-type]


@pytest.mark.vcr
def test_trace_config_hide_inputs(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    instrumentor = TogetherInstrumentor()
    instrumentor.uninstrument()
    instrumentor.instrument(tracer_provider=tracer_provider, config=TraceConfig(hide_inputs=True))
    client = Together()
    client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": "This is sensitive input. Reply with one word."}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert span.name == "Completions"
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.LLM.value
    assert attrs[SpanAttributes.LLM_PROVIDER] == OpenInferenceLLMProviderValues.TOGETHER.value
    assert attrs[SpanAttributes.LLM_MODEL_NAME] == _MODEL
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == "stop"
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert "sensitive" not in json.dumps({key: str(value) for key, value in attrs.items()})
    assert SpanAttributes.OUTPUT_VALUE in attrs


@pytest.mark.parametrize(
    "finish_reason",
    [
        "stop",
        "length",
        "tool_calls",
        "content_filter",
    ],
)
def test_finish_reason_values(
    finish_reason: str,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    in_memory_span_exporter.clear()

    with respx.mock(
        base_url="https://api.together.ai",
        assert_all_called=True,
    ) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=Response(
                status_code=200,
                json={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 1750000000,
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello!",
                            },
                            "finish_reason": finish_reason,
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 10,
                        "total_tokens": 15,
                    },
                },
            )
        )

        client = Together()
        client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": "Hello"}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert attrs[SpanAttributes.LLM_FINISH_REASON] == finish_reason
