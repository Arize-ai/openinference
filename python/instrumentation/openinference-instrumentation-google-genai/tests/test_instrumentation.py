# ruff: noqa: E501
from typing import Any, Dict, Iterator

import pytest
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.semconv.trace import MessageAttributes, SpanAttributes


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
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_generate_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = "REDACTED"

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Create content for the request
    content = Content(
        role="user",
        parts=[Part.from_text(text="What's the weather like?")],
    )

    # Create config
    config = GenerateContentConfig(
        system_instruction="You are a helpful assistant that can answer questions and help with tasks."
    )

    # Make the API call
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=content, config=config
    )

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Define expected attributes
    expected_attributes: Dict[str, Any] = {
        f"{SpanAttributes.LLM_PROVIDER}": "google",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "system",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "You are a helpful assistant that can answer questions and help with tasks.",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": "What's the weather like?",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": response.text,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Check if token counts are available in the response
    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        expected_attributes.update(
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response.usage_metadata.total_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response.usage_metadata.prompt_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: response.usage_metadata.candidates_token_count,
            }
        )

    # Verify attributes
    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value, (
            f"Attribute {key} does not match expected value"
        )


@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
@pytest.mark.asyncio
async def test_async_generate_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = "REDACTED"

    # Initialize the async client
    client = genai.Client(api_key=api_key).aio

    # Create content for the request
    content = Content(
        role="user",
        parts=[Part.from_text(text="What's the weather like?")],
    )

    # Create config
    config = GenerateContentConfig(
        system_instruction="You are a helpful assistant that can answer questions and help with tasks."
    )

    # Make the API call
    response = await client.models.generate_content(
        model="gemini-2.0-flash", contents=content, config=config
    )

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Define expected attributes
    expected_attributes: Dict[str, Any] = {
        f"{SpanAttributes.LLM_PROVIDER}": "google",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "system",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "You are a helpful assistant that can answer questions and help with tasks.",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": "What's the weather like?",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": response.text,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Check if token counts are available in the response
    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        expected_attributes.update(
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response.usage_metadata.total_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response.usage_metadata.prompt_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: response.usage_metadata.candidates_token_count,
            }
        )
    # Verify attributes
    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value, (
            f"Attribute {key} does not match expected value"
        )


@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_multi_turn_conversation(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = "REDACTED"

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Create a chat session
    chat = client.chats.create(model="gemini-2.0-flash")

    # Send first message
    response1 = chat.send_message("What is the capital of France?")

    # Send second message
    response2 = chat.send_message("Why is the sky blue?")

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2  # We should have two spans, one for each message

    # Check first span
    span1 = spans[0]
    attributes1 = dict(span1.attributes or {})

    expected_attributes1: Dict[str, Any] = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "What is the capital of France?",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": response1.text,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Check if token counts are available in the response
    if hasattr(response1, "usage_metadata") and response1.usage_metadata is not None:
        expected_attributes1.update(
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response1.usage_metadata.total_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response1.usage_metadata.prompt_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: response1.usage_metadata.candidates_token_count,
            }
        )

    # Verify attributes for first span
    for key, expected_value in expected_attributes1.items():
        assert attributes1.get(key) == expected_value, (
            f"Attribute {key} does not match expected value for first span"
        )

    # Check second span
    span2 = spans[1]
    attributes2 = dict(span2.attributes or {})

    expected_attributes2: Dict[str, Any] = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "What is the capital of France?",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": "The capital of France is **Paris**.\n",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_CONTENT}": "Why is the sky blue?",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": response2.text,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }
    # Check if token counts are available in the response
    if hasattr(response2, "usage_metadata") and response2.usage_metadata is not None:
        expected_attributes2.update(
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response2.usage_metadata.total_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response2.usage_metadata.prompt_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: response2.usage_metadata.candidates_token_count,
            }
        )

    # Verify attributes for second span in multi-turn conversation
    for key, expected_value in expected_attributes2.items():
        assert attributes2.get(key) == expected_value, (
            f"Attribute {key} does not match expected value for second span. Expected: {expected_value}, Actual: {attributes2.get(key)} key: {key}"
        )


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_streaming_text_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Initialize the client
    client = genai.Client(api_key="REDACTED")

    # Make the streaming API call
    stream = client.models.generate_content_stream(
        model="gemini-2.0-flash-001",
        contents=Content(
            role="user",
            parts=[Part.from_text(text="Tell me a short story about a cat.")],
        ),
    )

    # Collect all chunks from the stream
    full_response = ""
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
        full_response += chunk.text or ""

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Define expected attributes
    expected_attributes: Dict[str, Any] = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "Tell me a short story about a cat.",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash-001",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": full_response,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Check if token counts are available in the response. Complete usage metadata should be taken from the very last
    # chunk
    if chunks and hasattr(chunks[-1], "usage_metadata") and chunks[-1].usage_metadata is not None:
        expected_attributes.update(
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: chunks[-1].usage_metadata.total_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: chunks[-1].usage_metadata.prompt_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: chunks[
                    -1
                ].usage_metadata.candidates_token_count,
            }
        )

    # Verify attributes
    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value, (
            f"Attribute {key} does not match expected value"
        )


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
@pytest.mark.asyncio
async def test_async_streaming_text_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Initialize the async client
    client = genai.Client(api_key="REDACTED").aio

    # Make the streaming API call
    stream = await client.models.generate_content_stream(
        model="gemini-2.0-flash-001",
        contents=Content(
            role="user",
            parts=[Part.from_text(text="Tell me a short story about a cat within 20 words.")],
        ),
    )

    # Collect all chunks from the stream
    full_response = ""
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
        full_response += chunk.text or ""

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Define expected attributes
    expected_attributes: Dict[str, Any] = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "Tell me a short story about a cat within 20 words.",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash-001",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": full_response,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Check if token counts are available in the response. Complete usage metadata should be taken from the very last
    # chunk
    if chunks and hasattr(chunks[-1], "usage_metadata") and chunks[-1].usage_metadata is not None:
        expected_attributes.update(
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: chunks[-1].usage_metadata.total_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: chunks[-1].usage_metadata.prompt_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: chunks[
                    -1
                ].usage_metadata.candidates_token_count,
            }
        )

    # Verify attributes
    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value, (
            f"Attribute {key} does not match expected value"
        )
