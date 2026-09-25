import json
from typing import Any, Iterator, List

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
from openinference.instrumentation.together._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.semconv.trace import (
    AudioAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
    VideoAttributes,
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


def test_chat_with_multimodal_input(in_memory_span_exporter: InMemorySpanExporter) -> None:
    in_memory_span_exporter.clear()

    with respx.mock(
        base_url="https://api.together.ai",
        assert_all_called=True,
    ) as respx_mock:
        respx_mock.post("/v1/chat/completions").mock(
            return_value=Response(
                status_code=200,
                json={
                    "id": "chatcmpl-multimodal",
                    "object": "chat.completion",
                    "created": 1750000000,
                    "model": _MODEL,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "A cat meowing on a couch.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 8,
                        "total_tokens": 28,
                    },
                },
            )
        )

        client = Together()
        client.chat.completions.create(
            model=_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/cat.png"},
                        },
                        {
                            "type": "video_url",
                            "video_url": {"url": "https://example.com/cat.mp4"},
                        },
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "https://example.com/cat.mp3"},
                        },
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "ZmFrZQ==", "format": "wav"},
                        },
                    ],
                }
            ],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert (
        attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    )
    assert attrs.pop(SpanAttributes.LLM_PROVIDER) == OpenInferenceLLMProviderValues.TOGETHER.value
    assert attrs.pop(SpanAttributes.LLM_MODEL_NAME) == _MODEL
    assert attrs.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS) == json.dumps({"model": _MODEL})
    assert attrs.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attrs.pop(SpanAttributes.INPUT_VALUE)
    assert attrs.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert attrs.pop(SpanAttributes.OUTPUT_VALUE)
    assert attrs.pop(SpanAttributes.LLM_FINISH_REASON) == "stop"
    assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 20
    assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 8
    assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 28

    message = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert attrs.pop(f"{message}.{MessageAttributes.MESSAGE_ROLE}") == "user"
    contents = f"{message}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert attrs.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "text"
    assert (
        attrs.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "Describe this."
    )
    assert attrs.pop(f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "image"
    assert (
        attrs.pop(
            f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
            f"{ImageAttributes.IMAGE_URL}"
        )
        == "https://example.com/cat.png"
    )
    assert attrs.pop(f"{contents}.2.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "video"
    assert (
        attrs.pop(
            f"{contents}.2.{MessageContentAttributes.MESSAGE_CONTENT_VIDEO}."
            f"{VideoAttributes.VIDEO_URL}"
        )
        == "https://example.com/cat.mp4"
    )
    assert attrs.pop(f"{contents}.3.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "audio"
    assert (
        attrs.pop(
            f"{contents}.3.{MessageContentAttributes.MESSAGE_CONTENT_AUDIO}."
            f"{AudioAttributes.AUDIO_URL}"
        )
        == "https://example.com/cat.mp3"
    )
    assert attrs.pop(f"{contents}.4.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "audio"
    assert (
        attrs.pop(
            f"{contents}.4.{MessageContentAttributes.MESSAGE_CONTENT_AUDIO}."
            f"{AudioAttributes.AUDIO_URL}"
        )
        == "data:audio/wav;base64,ZmFrZQ=="
    )

    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attrs.pop(f"{output_message}.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert (
        attrs.pop(f"{output_message}.{MessageAttributes.MESSAGE_CONTENT}")
        == "A cat meowing on a couch."
    )
    assert not attrs


def test_message_content_as_tuple() -> None:
    # The SDKs accept any iterable of content parts, not only lists.
    message = {
        "role": "user",
        "content": (
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        ),
    }
    attributes = dict(_RequestAttributesExtractor()._get_attributes_from_message_param(message))
    assert attributes.pop(MessageAttributes.MESSAGE_ROLE) == "user"
    contents = MessageAttributes.MESSAGE_CONTENTS
    assert attributes.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "text"
    assert (
        attributes.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "What is in this image?"
    )
    assert (
        attributes.pop(f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}") == "image"
    )
    assert (
        attributes.pop(
            f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
            f"{ImageAttributes.IMAGE_URL}"
        )
        == "https://example.com/cat.png"
    )
    assert not attributes


def test_chat_with_generator_content(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    in_memory_span_exporter.clear()
    parts = (
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    )
    content: Iterator[Any] = (part for part in parts)

    with respx.mock(base_url="https://api.together.ai", assert_all_called=True) as respx_mock:
        route = respx_mock.post("/v1/chat/completions").mock(
            return_value=Response(
                status_code=200,
                json={
                    "id": "chatcmpl-generator",
                    "object": "chat.completion",
                    "created": 1750000000,
                    "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "A cat."},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        )
        client = Together()
        client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": content}],
        )

    # The generator is read once for the span, and the SDK still sends every part.
    assert json.loads(route.calls.last.request.content)["messages"][0]["content"] == list(parts)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    input_messages = {
        key: value
        for key, value in (spans[0].attributes or {}).items()
        if key.startswith(SpanAttributes.LLM_INPUT_MESSAGES)
    }
    message = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    assert input_messages.pop(f"{message}.{MessageAttributes.MESSAGE_ROLE}") == "user"
    contents = f"{message}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert (
        input_messages.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        input_messages.pop(f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}")
        == "What is in this image?"
    )
    assert (
        input_messages.pop(f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}")
        == "image"
    )
    assert (
        input_messages.pop(
            f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
            f"{ImageAttributes.IMAGE_URL}"
        )
        == "https://example.com/cat.png"
    )
    assert not input_messages
