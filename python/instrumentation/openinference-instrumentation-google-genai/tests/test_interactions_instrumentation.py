# type: ignore
# ruff: noqa: E501
from typing import Any, Dict, Iterator

import pytest
from google import genai
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    return exporter


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture
def setup_google_genai_instrumentation(tracer_provider: TracerProvider) -> Iterator[None]:
    instrumentor = GoogleGenAIInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    yield
    instrumentor.uninstrument()


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
@pytest.mark.parametrize("use_stream", [False, True])
def test_generate_interactions_simple_message(
    use_stream: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = "REDACTED"
    # Initialize the client
    client = genai.Client(api_key=api_key)
    input_message = "Tell me a short joke about programming."
    model_name = "gemini-3-flash-preview"
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
            if hasattr(chunk, "interaction") and chunk.interaction.usage:
                usage_metadata = chunk.interaction.usage
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
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "text",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}": "Because light attracts bugs.",
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
    assert attributes.pop(METADATA) is not None
    assert not attributes, f"Unexpected attributes found: {attributes}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
@pytest.mark.parametrize("use_stream", [False, True])
def test_generate_interactions_multi_model_messages(
    use_stream: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = "REDACTED"
    # Initialize the client
    client = genai.Client(api_key=api_key)
    input_messages = [
        {"type": "text", "text": "Describe the image."},
        {
            "type": "image",
            "uri": "https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U",
            "mime_type": "image/jpg",
        },
    ]
    model_name = "gemini-3-flash-preview"
    interaction = client.interactions.create(
        model=model_name,
        input=input_messages,
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
            if hasattr(chunk, "interaction") and chunk.interaction.usage:
                usage_metadata = chunk.interaction.usage
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
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}": "Describe the image.",
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}": "image",
        LLM_INVOCATION_PARAMETERS: '{"temperature": 0.7, "max_output_tokens": 500, "thinking_level": "low"}',
        LLM_MODEL_NAME: model_name,
        OUTPUT_MIME_TYPE: "text/plain",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}": "model",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "text",
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
    output_message = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
    )
    assert "black Labrador puppy" in output_message
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
    before_record_response=lambda _: {**_, "headers": {}},
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
    api_key = "REDACTED"

    # Initialize the client
    client = genai.Client(api_key=api_key).aio
    input_message = "Tell me a short joke about programming."
    model_name = "gemini-3-flash-preview"
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
            if hasattr(chunk, "interaction") and chunk.interaction.usage:
                usage_metadata = chunk.interaction.usage
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
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "text",
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}": "Because light attracts bugs.",
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
    assert attributes.pop(METADATA) is not None
    assert not attributes, f"Unexpected attributes found: {attributes}"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_agent_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = "REDACTED"
    # api_key = os.environ["GEMINI_API_KEY"]

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


LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
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
