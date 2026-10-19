# type: ignore
# ruff: noqa: E501
import asyncio
import gc
import os
from typing import Any, Dict

import pytest
import respx
from google import genai
from httpx import Response
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

TINY_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII="


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
)
@pytest.mark.parametrize("use_stream", [False, True])
def test_generate_interactions_simple_message(
    use_stream: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")
    # Initialize the client
    client = genai.Client(api_key=api_key)
    input_message = "Tell me a short joke about programming."
    model_name = "gemini-2.5-flash"
    interaction = client.interactions.create(
        model=model_name,
        input=input_message,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 500,
            "thinking_level": "low",
        },
        stream=use_stream,
    )
    usage_metadata = None
    if use_stream:
        for chunk in interaction:
            if (stream_interaction := getattr(chunk, "interaction", None)) and (
                usage := stream_interaction.usage
            ):
                usage_metadata = usage
    else:
        usage_metadata = interaction.usage
        assert interaction is not None

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    expected_attributes: Dict[str, Any] = {
        INPUT_MIME_TYPE: "text/plain",
        f"{LLM_PROVIDER}": "google",
        INPUT_VALUE: input_message,
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": "user",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}": input_message,
        LLM_INVOCATION_PARAMETERS: '{"temperature": 0.7, "max_output_tokens": 500, "thinking_level": "low"}',
        LLM_MODEL_NAME: model_name,
        OUTPUT_MIME_TYPE: "text/plain",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": "model",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "reasoning",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}": "text",
        OPENINFERENCE_SPAN_KIND: "LLM",
        LLM_TOKEN_COUNT_TOTAL: usage_metadata.total_tokens,
        LLM_TOKEN_COUNT_PROMPT: usage_metadata.total_input_tokens,
        LLM_TOKEN_COUNT_COMPLETION: usage_metadata.total_thought_tokens
        + usage_metadata.total_output_tokens,
    }
    for key, expected_value in expected_attributes.items():
        assert attributes.pop(key) == expected_value, (
            f"Attribute {key} does not match expected value: got {attributes.get(key)}"
        )
    assert attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_SIGNATURE}"
    )
    output_text = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}"
    )
    assert isinstance(output_text, str) and output_text
    assert attributes.pop(OUTPUT_VALUE) is not None
    assert attributes.pop(METADATA) is not None
    assert not attributes, f"Unexpected attributes found: {attributes}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
)
@pytest.mark.parametrize("use_stream", [False, True])
def test_generate_interactions_multi_model_messages(
    use_stream: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")
    # Initialize the client
    client = genai.Client(api_key=api_key)
    input_messages = [
        {"type": "text", "text": "Describe the image in one sentence."},
        {
            "type": "image",
            "data": TINY_PNG_BASE64,
            "mime_type": "image/png",
            "resolution": "low",
        },
    ]
    model_name = "gemini-2.5-flash-lite"
    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 64,
        "thinking_level": "high",
    }
    interaction = client.interactions.create(
        model=model_name,
        input=input_messages,
        generation_config=generation_config,
        stream=use_stream,
    )
    usage_metadata = None
    if use_stream:
        for chunk in interaction:
            if (stream_interaction := getattr(chunk, "interaction", None)) and (
                usage := stream_interaction.usage
            ):
                usage_metadata = usage
    else:
        usage_metadata = interaction.usage
        assert interaction is not None

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    expected_attributes: Dict[str, Any] = {
        INPUT_MIME_TYPE: "application/json",
        LLM_PROVIDER: "google",
        INPUT_VALUE: safe_json_dumps(input_messages),
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": "user",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "text",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}": "Describe the image in one sentence.",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}": "image",
        LLM_INVOCATION_PARAMETERS: safe_json_dumps(generation_config),
        LLM_MODEL_NAME: model_name,
        OUTPUT_MIME_TYPE: "text/plain",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": "model",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "reasoning",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}": "text",
        OPENINFERENCE_SPAN_KIND: "LLM",
        LLM_TOKEN_COUNT_TOTAL: usage_metadata.total_tokens,
        LLM_TOKEN_COUNT_PROMPT: usage_metadata.total_input_tokens,
        LLM_TOKEN_COUNT_COMPLETION: usage_metadata.total_thought_tokens
        + usage_metadata.total_output_tokens,
    }
    for key, expected_value in expected_attributes.items():
        assert attributes.pop(key) == expected_value, (
            f"Attribute {key} does not match expected value: got {attributes.get(key)}"
        )
    assert attributes.pop(OUTPUT_VALUE) is not None
    assert attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_SIGNATURE}"
    )
    output_message = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}"
    )
    assert isinstance(output_message, str) and output_message
    assert attributes.pop(METADATA) is not None
    assert (
        attributes.pop(
            f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"
        )
        is not None
    )
    assert not attributes, f"Unexpected attributes found: {attributes}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
)
@pytest.mark.parametrize("use_stream", [False, True])
@pytest.mark.asyncio
async def test_generate_interactions_async(
    use_stream: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key).aio
    input_message = "Tell me a short joke about programming."
    model_name = "gemini-2.5-flash"
    interaction = await client.interactions.create(
        model=model_name,
        input=input_message,
        generation_config={
            "temperature": 0.7,
            "max_output_tokens": 500,
            "thinking_level": "low",
        },
        stream=use_stream,
    )
    usage_metadata = None
    if use_stream:
        async for chunk in interaction:
            if (stream_interaction := getattr(chunk, "interaction", None)) and (
                usage := stream_interaction.usage
            ):
                usage_metadata = usage
    else:
        usage_metadata = interaction.usage
        assert interaction is not None

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    expected_attributes: Dict[str, Any] = {
        INPUT_MIME_TYPE: "text/plain",
        LLM_PROVIDER: "google",
        INPUT_VALUE: input_message,
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}": "user",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}": input_message,
        LLM_INVOCATION_PARAMETERS: '{"temperature": 0.7, "max_output_tokens": 500, "thinking_level": "low"}',
        LLM_MODEL_NAME: model_name,
        OUTPUT_MIME_TYPE: "text/plain",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": "model",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "reasoning",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}": "text",
        OPENINFERENCE_SPAN_KIND: "LLM",
        LLM_TOKEN_COUNT_TOTAL: usage_metadata.total_tokens,
        LLM_TOKEN_COUNT_PROMPT: usage_metadata.total_input_tokens,
        LLM_TOKEN_COUNT_COMPLETION: usage_metadata.total_thought_tokens
        + usage_metadata.total_output_tokens,
    }
    for key, expected_value in expected_attributes.items():
        assert attributes.pop(key) == expected_value, (
            f"Attribute {key} does not match expected value: got {attributes.get(key)}"
        )
    assert attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_SIGNATURE}"
    )
    output_text = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}"
    )
    assert isinstance(output_text, str) and output_text
    assert attributes.pop(OUTPUT_VALUE) is not None
    assert attributes.pop(METADATA) is not None
    assert not attributes, f"Unexpected attributes found: {attributes}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
)
def test_interactions_with_thinking_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")
    client = genai.Client(api_key=api_key)
    input_message = "Explain why the sky is blue in two sentences."
    model_name = "gemini-2.5-flash"
    generation_config = {
        "temperature": 0.7,
        "max_output_tokens": 1024,
        "thinking_level": "low",
        "thinking_summaries": "auto",
    }

    usage_metadata = None
    for chunk in client.interactions.create(
        model=model_name,
        input=input_message,
        generation_config=generation_config,
        stream=True,
    ):
        if (stream_interaction := getattr(chunk, "interaction", None)) and (
            usage := stream_interaction.usage
        ):
            usage_metadata = usage

    assert usage_metadata is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})

    # Reasoning content at index 0
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "reasoning"
    )
    reasoning_text = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}", None
    )
    assert isinstance(reasoning_text, str) and reasoning_text, (
        "reasoning summary text should be non-empty"
    )
    assert attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_SIGNATURE}"
    ), "reasoning signature should be present"

    # Answer text at index 1
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    output_text = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}"
    )
    assert isinstance(output_text, str) and output_text

    # Token counts
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == usage_metadata.total_tokens
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == usage_metadata.total_input_tokens
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == (
        usage_metadata.total_thought_tokens + usage_metadata.total_output_tokens
    )

    # Standard attributes
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "model"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(LLM_MODEL_NAME) == model_name
    assert attributes.pop(LLM_INVOCATION_PARAMETERS) == safe_json_dumps(generation_config)
    assert attributes.pop(LLM_PROVIDER) == "google"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(INPUT_VALUE) == input_message
    assert attributes.pop(INPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(OUTPUT_VALUE) is not None
    assert attributes.pop(METADATA) is not None
    assert not attributes, f"Unexpected attributes: {attributes}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
)
def test_agent_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    input_message = "Research the history of the Google TPUs with a focus on 2025 and 2026."
    client = genai.Client(api_key=api_key)
    interaction = client.interactions.create(
        input=input_message, agent="deep-research-pro-preview-12-2025", background=True
    )
    assert interaction.id is not None

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    expected_attributes: Dict[str, Any] = {
        INPUT_MIME_TYPE: "text/plain",
        INPUT_VALUE: input_message,
        OUTPUT_MIME_TYPE: "text/plain",
        OPENINFERENCE_SPAN_KIND: "AGENT",
    }
    for key, expected_value in expected_attributes.items():
        assert attributes.pop(key) == expected_value, (
            f"Attribute {key} does not match expected value: got {attributes.get(key)}"
        )
    assert attributes.pop(OUTPUT_VALUE) is not None
    assert not attributes, f"Unexpected attributes found: {attributes}"


# Partial SSE body for a streaming interactions.create() call: enough events to
# accumulate one model output chunk before the caller stops iterating.
_INTERACTIONS_STREAM_SSE = (
    "event: interaction.created\n"
    'data: {"interaction":{"id":"interaction-1","status":"in_progress","object":"interaction","model":"gemini-2.5-flash"},"event_type":"interaction.created"}\n'
    "\n"
    "event: step.start\n"
    'data: {"index":0,"step":{"type":"model_output"},"event_type":"step.start"}\n'
    "\n"
    "event: step.delta\n"
    'data: {"index":0,"delta":{"text":"hello ","type":"text"},"event_type":"step.delta"}\n'
    "\n"
    "event: step.delta\n"
    'data: {"index":0,"delta":{"text":"world","type":"text"},"event_type":"step.delta"}\n'
    "\n"
    "event: interaction.completed\n"
    'data: {"interaction":{"id":"interaction-1","status":"completed","created":"2026-10-19T00:00:00Z","updated":"2026-10-19T00:00:01Z","object":"interaction","model":"gemini-2.5-flash"},"event_type":"interaction.completed"}\n'
    "\n"
    "event: done\n"
    "data: [DONE]\n"
    "\n"
)


def _mock_interactions_stream(mock_router: Any) -> None:
    mock_router.post(path__regex=r"/v1beta/interactions.*").mock(
        return_value=Response(
            200, text=_INTERACTIONS_STREAM_SSE, headers={"content-type": "text/event-stream"}
        )
    )


def _assert_interactions_span_ended_with_partial_output(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.status_code == StatusCode.UNSET
    assert [event.name for event in span.events] == []
    attributes = dict(span.attributes or {})
    assert (
        attributes.get(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == "hello "
    )


@pytest.mark.parametrize("abandon", ["break", "close", "drop"])
def test_interactions_stream_left_early_ends_span(
    abandon: str,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    client = genai.Client(api_key="fake-key")
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock_router:
        _mock_interactions_stream(mock_router)
        stream = client.interactions.create(model="gemini-2.5-flash", input="hi", stream=True)
        if abandon == "break":
            for event in stream:
                if getattr(event, "event_type", None) == "step.delta":
                    break
        else:
            iterator = iter(stream)
            for _ in range(3):
                next(iterator)
            if abandon == "close":
                iterator.close()
            else:
                del iterator
                gc.collect()

    _assert_interactions_span_ended_with_partial_output(in_memory_span_exporter)


@pytest.mark.parametrize("abandon", ["break", "aclose"])
async def test_async_interactions_stream_left_early_ends_span(
    abandon: str,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    client = genai.Client(api_key="fake-key").aio
    with respx.mock(base_url="https://generativelanguage.googleapis.com") as mock_router:
        _mock_interactions_stream(mock_router)
        stream = await client.interactions.create(model="gemini-2.5-flash", input="hi", stream=True)
        if abandon == "break":
            async for event in stream:
                if getattr(event, "event_type", None) == "step.delta":
                    break
            gc.collect()
            for _ in range(3):
                await asyncio.sleep(0)
        else:
            iterator = stream.__aiter__()
            for _ in range(3):
                await iterator.__anext__()
            await iterator.aclose()

    _assert_interactions_span_ended_with_partial_output(in_memory_span_exporter)


LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_SIGNATURE = MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
IMAGE_URL = ImageAttributes.IMAGE_URL
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
INPUT_VALUE = SpanAttributes.INPUT_VALUE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
METADATA = SpanAttributes.METADATA
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
