# type: ignore
# ruff: noqa: E501
import json
import os
from typing import Any, Dict

import pytest
from google import genai
from google.genai import types
from google.genai.types import (
    Content,
    EmbedContentConfig,
    FunctionCall,
    FunctionDeclaration,
    FunctionResponse,
    GenerateContentConfig,
    Part,
    Tool,
    ToolCodeExecution,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


class Answer(BaseModel):
    answer: str


@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_embed_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Create content for the request
    content = Content(
        parts=[
            Part.from_text(text="Why is the sky blue?"),
            Part.from_text(text="What is the capital of France?"),
        ],
    )

    # Create config
    config = EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")

    # Make the API call
    response = client.models.embed_content(
        model="gemini-embedding-001", contents=content, config=config
    )
    assert response is not None

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Verify expected attributes
    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "EMBEDDING"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "google"
    assert attributes.pop(SpanAttributes.EMBEDDING_MODEL_NAME) == "gemini-embedding-001"
    assert (
        attributes.pop(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "Why is the sky blue?\n\nWhat is the capital of France?"
    )
    # Verify embedding vectors are present
    assert (
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
        in attributes
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) is not None
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS) == json.dumps(
        {"task_type": "RETRIEVAL_DOCUMENT"}
    )


@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_embed_content_multiple_contents(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    """Test embedding multiple Content objects — each should get its own vector."""
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")
    client = genai.Client(api_key=api_key)

    # Pass a list of strings — each becomes a separate Content / embedding
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=["Why is the sky blue?", "What is the capital of France?"],
    )
    assert response is not None

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "EMBEDDING"
    assert attributes.pop(SpanAttributes.EMBEDDING_MODEL_NAME) == "gemini-embedding-001"
    # Two separate EMBEDDING_TEXT entries — one per content string
    assert (
        attributes.pop(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "Why is the sky blue?"
    )
    assert (
        attributes.pop(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.1.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "What is the capital of France?"
    )


@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
@pytest.mark.asyncio
async def test_async_embed_content(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the async client
    client = genai.Client(api_key=api_key).aio

    # Create content for the request
    content = Content(
        parts=[
            Part.from_text(text="Why is the sky blue?"),
            Part.from_text(text="What is the capital of France?"),
        ],
    )

    # Create config
    config = EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")

    # Make the API call
    response = await client.models.embed_content(
        model="gemini-embedding-001", contents=content, config=config
    )
    assert response is not None

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]

    # Verify expected attributes
    attributes = dict(span.attributes or {})
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "EMBEDDING"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "google"
    assert attributes.pop(SpanAttributes.EMBEDDING_MODEL_NAME) == "gemini-embedding-001"
    assert (
        attributes.pop(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "Why is the sky blue?\n\nWhat is the capital of France?"
    )
    # Verify embedding vectors are present
    assert (
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
        in attributes
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) is not None
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS) == json.dumps(
        {"task_type": "RETRIEVAL_DOCUMENT"}
    )


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
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Create content for the request
    contents = [
        Content(
            role="user",
            parts=[
                Part.from_text(text="What's the weather like?"),
            ],
        ),
        Content(
            role="model",
            parts=[
                Part(
                    function_call=FunctionCall(
                        name="get_weather", args={"location": "San Francisco"}, id="call_abc123"
                    )
                ),
            ],
        ),
        Content(
            role="user",
            parts=[
                Part(
                    function_response=FunctionResponse(
                        name="get_weather",
                        response={
                            "location": "San Francisco",
                            "temperature": 65,
                            "unit": "fahrenheit",
                            "condition": "foggy",
                            "humidity": "85%",
                        },
                        id="call_abc123",
                    )
                ),
            ],
        ),
    ]

    # Create config
    config = GenerateContentConfig(
        system_instruction="You are a helpful assistant that can answer questions and help with tasks."
    )

    # Make the API call
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=contents, config=config
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
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}": "get_weather",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}": json.dumps(
            {"location": "San Francisco"}
        ),
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.2.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_ID}": "call_abc123",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.3.{MessageAttributes.MESSAGE_TOOL_CALL_ID}": "call_abc123",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.3.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.3.{MessageAttributes.MESSAGE_CONTENT}": json.dumps(
            {
                "location": "San Francisco",
                "temperature": 65,
                "unit": "fahrenheit",
                "condition": "foggy",
                "humidity": "85%",
            }
        ),
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
    before_record_response=lambda r: {
        **r,
        "headers": {
            k: v
            for k, v in r["headers"].items()
            if k.lower() in ("content-encoding", "content-type")
        },
    },
)
@pytest.mark.parametrize("streaming", [False, True])
def test_generate_content_describe_image(
    streaming: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = "REDACTED"

    # Initialize the client
    client = genai.Client(api_key=api_key)

    config = GenerateContentConfig(
        system_instruction=(
            "You are a helpful assistant that can answer questions and help with tasks."
        )
    )
    image_bytes = b"iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQMAAAD+wSzIAAAABlBMVEX///+/v7+jQ3Y5AAAADklEQVQI12P4AIX8EAgALgAD/aNpbtEAAAAASUVORK5CYII"
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
    content = Content(
        role="user",
        parts=[
            Part.from_text(text="Describe Image."),
            image_part,
        ],
    )
    if streaming:
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=content,
            config=config,
        )
        for res in response:
            ...
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content,
            config=config,
        )
        assert response.text

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
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}": "Describe Image.",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}": "text",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}": "image",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.5-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Verify attributes
    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value, (
            f"Attribute {key} does not match expected value"
        )
    key = f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
    assert attributes.get(key), "Image Url should be present in span attributes"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_generate_content_with_config_as_dict(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Create content for the request
    content = Content(
        role="user",
        parts=[Part.from_text(text="Repeat: this is a test")],
    )

    # Create config
    config = {
        "temperature": 0.5,
        "top_p": 0.95,
        "top_k": 40,
        "candidate_count": 1,
        "thinking_config": {"thinking_budget": 100},
    }

    # # Make the API call
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=content, config=config
    )

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Define expected attributes
    expected_attributes: Dict[str, Any] = {
        f"{SpanAttributes.LLM_PROVIDER}": "google",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "Repeat: this is a test",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.5-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": response.text,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
        SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps(
            {
                "temperature": 0.5,
                "top_p": 0.95,
                "top_k": 40.0,
                "candidate_count": 1,
                "thinking_config": {"thinking_budget": 100},
            }
        ),
    }

    # Check if token counts are available in the response
    if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
        completion_token_count = 0
        if candidates := response.usage_metadata.candidates_token_count:
            completion_token_count += candidates
        if thoughts := response.usage_metadata.thoughts_token_count:
            completion_token_count += thoughts

        expected_attributes.update(
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: response.usage_metadata.total_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: response.usage_metadata.prompt_token_count,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: completion_token_count,
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
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

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
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

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
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "REDACTED"))

    # Make the streaming API call
    stream = client.models.generate_content_stream(
        model="gemini-2.0-flash",
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
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
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
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "REDACTED")).aio

    # Make the streaming API call
    stream = await client.models.generate_content_stream(
        model="gemini-2.0-flash",
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
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
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
def test_generate_content_with_tool(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # REDACT API Key, Cassette has stored response, delete cassette and replace API Key to edit test
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Define a tool/function for weather information
    weather_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="get_weather",
                description="Get current weather information for a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country for weather information",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ]
    )

    # Create content for the request
    user_message = "What's the weather like in San Francisco?"
    content = Content(
        role="user",
        parts=[Part.from_text(text=user_message)],
    )

    # Create config with tools
    system_instruction = "You are a helpful assistant that can answer questions and help with tasks. Use the available tools when appropriate."
    config = GenerateContentConfig(system_instruction=system_instruction, tools=[weather_tool])

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
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": system_instruction,
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": user_message,
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": response.text
        or None,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Verify flattened tool schema format
    tool_schema_key = f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"
    assert tool_schema_key in attributes, "Tool schema not found in attributes"
    tool_schema_json = attributes.get(tool_schema_key)
    assert isinstance(tool_schema_json, str), "Tool schema should be a JSON string"
    tool_schema = json.loads(tool_schema_json)

    # Verify tool matches flattened format
    expected_tool_schema = {
        "name": "get_weather",
        "description": "Get current weather information for a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state/country for weather information",
                },
                "unit": {
                    "type": "STRING",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    }

    assert tool_schema == expected_tool_schema, (
        f"Tool schema doesn't match expected flattened format.\n"
        f"Expected: {expected_tool_schema}\n"
        f"Got: {tool_schema}"
    )

    # Check if the model decided to call the tool
    tool_call_name_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    tool_call_args_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"

    if tool_call_name_key in attributes:
        # Model decided to call the tool, verify the tool call details
        assert attributes.get(tool_call_name_key) == "get_weather", (
            "Expected tool call to be 'get_weather'"
        )

        tool_call_args = attributes.get(tool_call_args_key)
        assert isinstance(tool_call_args, str), "Tool call arguments should be a JSON string"

        # Parse and validate tool call arguments
        args = json.loads(tool_call_args)
        assert "location" in args, "Tool call should include 'location' parameter"
        # The location should be something reasonable for San Francisco
        assert "san francisco" in args["location"].lower() or "sf" in args["location"].lower(), (
            "Tool call location should reference San Francisco"
        )

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
            f"Attribute {key} does not match expected value\n"
            f"Expected: {expected_value}\n"
            f"Got: {attributes.get(key)}"
        )


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_generate_content_with_raw_json_tool(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # REDACT API Key, Cassette has stored response, delete cassette and replace API Key to edit test
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Define a tool/function using raw JSON instead of Pydantic objects
    weather_tool_dict = {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "Get current weather information for a given location",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "location": {
                            "type": "STRING",
                            "description": "The city and state/country for weather information",
                        },
                        "unit": {
                            "type": "STRING",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            }
        ]
    }

    # Create content for the request
    user_message = "What's the weather like in San Francisco?"
    content = Content(
        role="user",
        parts=[Part.from_text(text=user_message)],
    )

    # Create config with raw JSON tools
    system_instruction = "You are a helpful assistant that can answer questions and help with tasks. Use the available tools when appropriate."
    config = GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[weather_tool_dict],  # Using raw dict instead of Tool object
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
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": system_instruction,
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": user_message,
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": response.text
        or None,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Verify tool schema is recorded
    tool_schema_key = f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"
    assert tool_schema_key in attributes, "Tool schema not found in attributes"
    tool_schema_json = attributes.get(tool_schema_key)
    assert isinstance(tool_schema_json, str), "Tool schema should be a JSON string"

    # Parse and validate the tool schema matches what we provided
    tool_schema = json.loads(tool_schema_json)
    # For raw JSON tools, the expected schema should be flattened
    expected_tool_schema = {
        "name": "get_weather",
        "description": "Get current weather information for a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state/country for weather information",
                },
                "unit": {
                    "type": "STRING",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    }
    assert tool_schema == expected_tool_schema, (
        f"Tool schema does not match expected schema. Expected: {expected_tool_schema}, Got: {tool_schema}"
    )

    # Check if the model decided to call the tool
    tool_call_name_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    tool_call_args_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"

    if tool_call_name_key in attributes:
        # Model decided to call the tool, verify the tool call details
        assert attributes.get(tool_call_name_key) == "get_weather", (
            "Expected tool call to be 'get_weather'"
        )

        tool_call_args = attributes.get(tool_call_args_key)
        assert isinstance(tool_call_args, str), "Tool call arguments should be a JSON string"

        # Parse and validate tool call arguments
        args = json.loads(tool_call_args)
        assert "location" in args, "Tool call should include 'location' parameter"
        # The location should be something reasonable for San Francisco
        assert "san francisco" in args["location"].lower() or "sf" in args["location"].lower(), (
            "Tool call location should reference San Francisco"
        )

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
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_streaming_content_with_tool(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # REDACT API Key, Cassette has stored response, delete cassette and replace API Key to edit test
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Define a tool/function for weather information
    weather_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="get_weather",
                description="Get current weather information for a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country for weather information",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            )
        ]
    )

    # Create content for the request
    user_message = "What's the weather like in San Francisco?"
    content = Content(
        role="user",
        parts=[Part.from_text(text=user_message)],
    )

    # Create config with tools
    system_instruction = "You are a helpful assistant that can answer questions and help with tasks. Use the available tools when appropriate."
    config = GenerateContentConfig(system_instruction=system_instruction, tools=[weather_tool])

    # Make the streaming API call
    stream = client.models.generate_content_stream(
        model="gemini-2.0-flash", contents=content, config=config
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
        f"{SpanAttributes.LLM_PROVIDER}": "google",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "system",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": system_instruction,
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": user_message,
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Only add message content if there was actual text (not just tool calls)
    if full_response:
        expected_attributes[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
        ] = full_response

    # Verify tool schema is recorded (same as non-streaming)
    tool_schema_key = f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"
    assert tool_schema_key in attributes, "Tool schema not found in attributes"
    tool_schema_json = attributes.get(tool_schema_key)
    assert isinstance(tool_schema_json, str), "Tool schema should be a JSON string"

    # Parse and validate the tool schema matches what we provided
    tool_schema = json.loads(tool_schema_json)

    # Verify tool matches flattened format
    expected_tool_schema = {
        "name": "get_weather",
        "description": "Get current weather information for a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state/country for weather information",
                },
                "unit": {
                    "type": "STRING",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    }
    assert tool_schema == expected_tool_schema, (
        f"Tool schema does not match expected schema. Expected: {expected_tool_schema}, Got: {tool_schema}"
    )

    # Check if the model decided to call the tool in streaming response
    tool_call_name_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    tool_call_args_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"

    # For this test, we expect a tool call since we're testing tool calling functionality
    assert tool_call_name_key in attributes, (
        f"Expected a tool call in the streaming response, but none found. Available keys: {list(attributes.keys())}"
    )

    # Model decided to call the tool, verify the tool call details
    assert attributes.get(tool_call_name_key) == "get_weather", (
        "Expected tool call to be 'get_weather'"
    )

    tool_call_args = attributes.get(tool_call_args_key)
    assert isinstance(tool_call_args, str), "Tool call arguments should be a JSON string"

    # Parse and validate tool call arguments
    args = json.loads(tool_call_args)
    assert "location" in args, "Tool call should include 'location' parameter"
    # The location should be something reasonable for San Francisco
    assert "san francisco" in args["location"].lower() or "sf" in args["location"].lower(), (
        "Tool call location should reference San Francisco"
    )

    # Check if token counts are available in the response from the last chunk
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
@pytest.mark.parametrize("streaming", [False, True])
def test_response_with_multiple_tool_calls(
    streaming: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    api_key = "dummy-key"

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Define a tool/function for weather information
    weather_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="get_weather",
                description="Get current weather information for a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country for weather information",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            )
        ]
    )

    # Create content for the request
    user_message = "What is the weather like in Boston & new Delhi?"
    content = Content(
        role="user",
        parts=[Part.from_text(text=user_message)],
    )

    # Create config with tools
    system_instruction = "You are a helpful assistant that can answer questions and help with tasks. Use the available tools when appropriate."
    config = GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[weather_tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Make the streaming API call
    if streaming:
        response = client.models.generate_content_stream(
            model="gemini-2.0-flash", contents=content, config=config
        )
        for rec in response:
            ...
        # Collect all chunks from the stream
        full_response = ""
    else:
        response = client.models.generate_content(
            model="gemini-2.0-flash", contents=content, config=config
        )
        # Collect all chunks from the stream
        full_response = response.text or ""

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Define expected attributes
    expected_attributes: Dict[str, Any] = {
        f"{SpanAttributes.LLM_PROVIDER}": "google",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "system",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": system_instruction,
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": user_message,
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Only add message content if there was actual text (not just tool calls)
    if full_response:
        expected_attributes[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
        ] = full_response

    # Verify tool schema is recorded (same as non-streaming)
    tool_schema_key = f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"
    assert tool_schema_key in attributes, "Tool schema not found in attributes"
    tool_schema_json = attributes.get(tool_schema_key)
    assert isinstance(tool_schema_json, str), "Tool schema should be a JSON string"

    # Parse and validate the tool schema matches what we provided
    tool_schema = json.loads(tool_schema_json)

    # Verify tool matches flattened format
    expected_tool_schema = {
        "name": "get_weather",
        "description": "Get current weather information for a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state/country for weather information",
                },
                "unit": {
                    "type": "STRING",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    }
    assert tool_schema == expected_tool_schema, (
        f"Tool schema does not match expected schema. Expected: {expected_tool_schema}, Got: {tool_schema}"
    )

    expected_locations = ["boston", "new delhi"]
    for i in range(2):
        # Check if the model decided to call the tool in streaming response
        tool_call_name_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{i}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
        tool_call_args_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.{i}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"

        # For this test, we expect a tool call since we're testing tool calling functionality
        assert tool_call_name_key in attributes, (
            f"Expected a tool call in the streaming response, but none found. Available keys: {list(attributes.keys())}"
        )

        # Model decided to call the tool, verify the tool call details
        assert attributes.get(tool_call_name_key) == "get_weather", (
            "Expected tool call to be 'get_weather'"
        )

        tool_call_args = attributes.get(tool_call_args_key)
        assert isinstance(tool_call_args, str), "Tool call arguments should be a JSON string"

        # Parse and validate tool call arguments
        args = json.loads(tool_call_args)
        assert "location" in args, "Tool call should include 'location' parameter"
        # The location should be something reasonable for San Francisco
        assert expected_locations[i] in args["location"].lower(), (
            "Tool call location should reference San Francisco"
        )

    # Check if token counts are available in the response from the last chunk
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
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_chat_session_with_tool(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # REDACT API Key, Cassette has stored response, delete cassette and replace API Key to edit test
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Define a tool/function for weather information
    weather_tool = Tool(
        function_declarations=[
            FunctionDeclaration(
                name="get_weather",
                description="Get current weather information for a given location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state/country for weather information",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            )
        ]
    )

    # Create config with tools for the chat session
    user_message = "What's the weather like in San Francisco?"
    system_instruction = "You are a helpful assistant that can answer questions and help with tasks. Use the available tools when appropriate."

    config = GenerateContentConfig(system_instruction=system_instruction, tools=[weather_tool])

    # Create a chat session with tools
    chat = client.chats.create(model="gemini-2.0-flash", config=config)

    # Send message without tools config (tools are already configured for the chat session)
    response = chat.send_message(user_message)

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Define expected attributes (following Anthropic pattern)
    expected_attributes: Dict[str, Any] = {
        f"{SpanAttributes.LLM_PROVIDER}": "google",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "system",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": system_instruction,
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": user_message,
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # Only add message content if there was actual text (tool calls might not have text content)
    if response.text:
        expected_attributes[
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
        ] = response.text

    # Verify tool schema is recorded
    tool_schema_key = f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"
    assert tool_schema_key in attributes, "Tool schema not found in attributes"
    tool_schema_json = attributes.get(tool_schema_key)
    assert isinstance(tool_schema_json, str), "Tool schema should be a JSON string"

    # Parse and validate the tool schema matches what we provided
    tool_schema = json.loads(tool_schema_json)

    # Verify tool matches flattened format
    expected_tool_schema = {
        "name": "get_weather",
        "description": "Get current weather information for a given location",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "location": {
                    "type": "STRING",
                    "description": "The city and state/country for weather information",
                },
                "unit": {
                    "type": "STRING",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    }
    assert tool_schema == expected_tool_schema, (
        f"Tool schema does not match expected schema. Expected: {expected_tool_schema}, Got: {tool_schema}"
    )

    # Check if the model decided to call the tool
    tool_call_name_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    tool_call_args_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"

    # For chat session tool calling, we expect a tool call since we're testing tool calling functionality
    assert tool_call_name_key in attributes, (
        f"Expected a tool call in the chat session response, but none found. Available keys: {list(attributes.keys())}"
    )

    # Model decided to call the tool, verify the tool call details
    assert attributes.get(tool_call_name_key) == "get_weather", (
        "Expected tool call to be 'get_weather'"
    )

    tool_call_args = attributes.get(tool_call_args_key)
    assert isinstance(tool_call_args, str), "Tool call arguments should be a JSON string"

    # Parse and validate tool call arguments
    args = json.loads(tool_call_args)
    assert "location" in args, "Tool call should include 'location' parameter"
    # The location should be something reasonable for San Francisco
    assert "san francisco" in args["location"].lower() or "sf" in args["location"].lower(), (
        "Tool call location should reference San Francisco"
    )

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


# This test may be overkill, just want to make sure the logic in the accumulator works as expected. Tried to mock out
# Chunks for client to return, but ultimately could not get that approach to work
def test_streaming_tool_call_aggregation(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    """Test that streaming tool calls are properly aggregated across multiple chunks."""
    # Direct test of the streaming accumulator logic
    from openinference.instrumentation.google_genai._stream import (
        _ResponseAccumulator,
        _ResponseExtractor,
    )
    from openinference.semconv.trace import MessageAttributes, SpanAttributes, ToolCallAttributes

    # Create a response accumulator (this is what processes streaming chunks)
    accumulator = _ResponseAccumulator()

    # Create mock chunks that simulate a tool call arriving in parts
    class MockChunk:
        def __init__(self, data):
            self.data = data

        def model_dump(self, exclude_unset=True, warnings=False):
            return self.data

    # Chunk 1: Function name only
    chunk1 = MockChunk(
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"function_call": {"name": "get_weather"}}],
                    },
                }
            ],
            "model_version": "gemini-2.0-flash",
        }
    )

    # Chunk 2: Function arguments only
    chunk2 = MockChunk(
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "args": {"location": "San Francisco", "unit": "fahrenheit"}
                                }
                            }
                        ],
                    },
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 50,
                "candidates_token_count": 10,
                "total_token_count": 60,
            },
        }
    )

    # Process chunks through the accumulator (this tests our aggregation logic)
    accumulator.process_chunk(chunk1)
    accumulator.process_chunk(chunk2)

    extractor = _ResponseExtractor(accumulator)
    attributes = dict(extractor.get_attributes())

    # Verify the aggregated tool call - this is the key test!
    tool_call_name_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    tool_call_args_key = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"

    # This tests that both name (from chunk1) and args (from chunk2) are present
    assert attributes.get(tool_call_name_key) == "get_weather", (
        f"Function name not found. Available keys: {list(attributes.keys())}"
    )

    tool_call_args = attributes.get(tool_call_args_key)
    assert tool_call_args is not None, (
        f"Function arguments not found. Available keys: {list(attributes.keys())}"
    )

    # Parse and verify the merged arguments
    args = json.loads(tool_call_args)
    assert args["location"] == "San Francisco", (
        "Location argument missing from aggregated tool call"
    )
    assert args["unit"] == "fahrenheit", "Unit argument missing from aggregated tool call"
    assert (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.1.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
        not in attributes
    )
    assert (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.1.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        not in attributes
    )

    # Verify token counts from final chunk
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 60
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 50
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 10


def test_streaming_multimodal_content_aggregation() -> None:
    """Test that streamed text and image parts preserve their original positions."""
    from openinference.instrumentation.google_genai._stream import (
        _ResponseAccumulator,
        _ResponseExtractor,
    )

    class MockChunk:
        def __init__(self, data):
            self.data = data

        def model_dump(self, exclude_unset=True, warnings=False):
            return self.data

    accumulator = _ResponseAccumulator()
    accumulator.process_chunk(
        MockChunk(
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "caption"},
                                {"inline_data": {"mime_type": "image/png", "data": b"img1"}},
                                {"inline_data": {"mime_type": "image/png", "data": b"img2"}},
                            ],
                        },
                    }
                ],
                "model_version": "gemini-2.5-flash-image",
            }
        )
    )

    attributes = dict(_ResponseExtractor(accumulator).get_attributes())
    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"

    assert attributes.get(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}") == "model"
    assert f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}" not in attributes
    assert (
        attributes.get(
            f"{prefix}.{MessageAttributes.MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"
        )
        == "caption"
    )
    assert (
        attributes.get(
            f"{prefix}.{MessageAttributes.MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"
        )
        == "text"
    )
    for index, encoded in ((1, "aW1nMQ=="), (2, "aW1nMg==")):
        image_url_key = (
            f"{prefix}.{MessageAttributes.MESSAGE_CONTENTS}.{index}."
            f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
        )
        type_key = (
            f"{prefix}.{MessageAttributes.MESSAGE_CONTENTS}.{index}."
            f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"
        )
        assert attributes.get(image_url_key) == f"data:image/png;base64,{encoded}"
        assert attributes.get(type_key) == "image"


def test_response_attributes_extract_image_from_file_data() -> None:
    from openinference.instrumentation.google_genai._response_attributes_extractor import (
        _ResponseAttributesExtractor,
    )

    file_uri = "https://example.com/cat.jpg"
    response = types.GenerateContentResponse.model_validate(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"file_data": {"mime_type": "image/jpeg", "file_uri": file_uri}}],
                    }
                }
            ],
            "model_version": "gemini-test",
        }
    )

    attributes = dict(_ResponseAttributesExtractor().get_attributes(response, {}))
    image_url_key = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
        f"{MessageAttributes.MESSAGE_CONTENTS}.0."
        f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
    )
    type_key = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0."
        f"{MessageAttributes.MESSAGE_CONTENTS}.0."
        f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"
    )
    assert attributes.get(image_url_key) == file_uri
    assert attributes.get(type_key) == "image"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_generate_content_with_automatic_tool_calling(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    """Test automatic tool calling where Google GenAI executes the function and returns complete response."""
    # Get API key from environment variable
    api_key = os.environ.get("GEMINI_API_KEY", "REDACTED")

    # Initialize the client
    client = genai.Client(api_key=api_key)

    # Define an executable function for automatic tool calling
    def get_weather(location: str, unit: str = "fahrenheit") -> Dict[str, Any]:
        """
        This function will be automatically executed by Google GenAI.
        It must return a JSON-serializable value.
        """
        # Mock weather data for testing
        mock_weather = {
            "san francisco": {"temperature": 65, "condition": "foggy", "humidity": "85%"},
            "new york": {"temperature": 72, "condition": "sunny", "humidity": "60%"},
            "london": {"temperature": 55, "condition": "rainy", "humidity": "90%"},
        }

        location_lower = location.lower()
        weather = mock_weather.get(
            location_lower, {"temperature": 70, "condition": "unknown", "humidity": "50%"}
        )

        if unit == "celsius":
            weather["temperature"] = round((weather["temperature"] - 32) * 5 / 9, 1)

        return {
            "location": location,
            "temperature": weather["temperature"],
            "unit": unit,
            "condition": weather["condition"],
            "humidity": weather["humidity"],
        }

    # Create content for the request
    user_message = "What's the weather like in San Francisco?"
    content = Content(
        role="user",
        parts=[Part.from_text(text=user_message)],
    )

    # Create config with executable function (automatic tool calling)
    # According to Google GenAI docs, pass the function directly in tools, not wrapped in Tool objects
    system_instruction = "You are a helpful assistant that can answer questions and help with tasks. Use the available tools when appropriate."
    config = GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[get_weather],  # Pass actual function directly for automatic calling!
    )

    # Make the API call
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=content, config=config
    )

    # For automatic tool calling, we expect a complete text response
    # The model should have automatically called the function and incorporated the results
    assert response is not None, "Response should not be None"
    assert response.text is not None, "Response should have text content for automatic tool calling"
    assert len(response.text.strip()) > 0, "Response text should not be empty"

    # Validate that the response contains weather information
    response_lower = response.text.lower()
    weather_keywords = ["weather", "temperature", "degrees", "san francisco", "condition"]
    assert any(keyword in response_lower for keyword in weather_keywords), (
        f"Response should contain weather-related information. Got: '{response.text}'"
    )

    # Get the spans
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Define expected attributes
    expected_attributes: Dict[str, Any] = {
        f"{SpanAttributes.LLM_PROVIDER}": "google",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "system",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": system_instruction,
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}": user_message,
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.0-flash",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": response.text,
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
    }

    # For automatic tool calling, the tool schema should still be recorded
    # but it will be auto-generated from the function signature
    tool_schema_key = f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"
    assert tool_schema_key in attributes, "Tool schema not found in attributes"
    tool_schema_json = attributes.get(tool_schema_key)
    assert isinstance(tool_schema_json, str), "Tool schema should be a JSON string"

    # Parse and validate the tool schema was auto-generated correctly
    tool_schema = json.loads(tool_schema_json)

    # For automatic function calling, the tool should be in flattened format
    assert "name" in tool_schema, "Tool schema should have name"
    assert tool_schema["name"] == "get_weather", "Function name should match"
    assert "parameters" in tool_schema, "Function should have parameters"

    # Verify key parameters are present
    params = tool_schema["parameters"]
    assert "properties" in params, "Parameters should have properties"
    assert "location" in params["properties"], "Should have location parameter"

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

    # Note: For automatic tool calling, the function execution happens transparently
    # We may or may not see explicit tool call attributes in the span depending on
    # how Google GenAI implements it internally. The key difference is that we get
    # a complete text response that incorporates the function results.


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_validate_token_counts(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "REDACTED"))
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="What is the sum of the first 50 prime numbers? "
        "Generate and run code for the calculation, and make sure you get all 50.",
        config=GenerateContentConfig(tools=[Tool(code_execution=ToolCodeExecution)]),
    )
    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    attributes = dict(span.attributes or {})

    usage_metadata = response.usage_metadata
    assert usage_metadata is not None, "Expected usage metadata to be present"
    prompt_token_count = 0
    if usage_metadata.prompt_token_count:
        prompt_token_count += usage_metadata.prompt_token_count
    if usage_metadata.tool_use_prompt_token_count:
        prompt_token_count += usage_metadata.tool_use_prompt_token_count
    completion_token_count = 0
    if usage_metadata.candidates_token_count:
        completion_token_count += usage_metadata.candidates_token_count
    if usage_metadata.thoughts_token_count:
        completion_token_count += usage_metadata.thoughts_token_count
    expected_attributes = {}
    if usage_metadata.total_token_count:
        expected_attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = usage_metadata.total_token_count
    if prompt_token_count:
        expected_attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = prompt_token_count
    if completion_token_count:
        expected_attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = completion_token_count
    if usage_metadata.thoughts_token_count:
        expected_attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] = (
            usage_metadata.thoughts_token_count
        )
    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value, (
            f"Attribute {key} does not match expected value: got {attributes.get(key)}"
        )


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
@pytest.mark.parametrize("streaming", [False, True])
def test_generate_content_with_file_uri_image(
    streaming: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    """Test that file_data (URI-based images) are captured as span attributes.

    Uses Part.from_uri which sends fileData in the request body instead of
    inlineData bytes, allowing the model to fetch the image from the given URI.
    Delete the cassette and set GEMINI_API_KEY to re-record.
    """
    api_key = "api-key-placeholder"
    client = genai.Client(api_key=api_key)

    file_uri = "https://fastly.picsum.photos/id/237/200/300.jpg?hmac=TmmQSbShHz9CdQm0NkEjx1Dyh_Y984R9LpNrpvH2D_U"
    content = Content(
        role="user",
        parts=[
            Part.from_text(text="Describe this image in one sentence."),
            Part.from_uri(file_uri=file_uri, mime_type="image/jpeg"),
        ],
    )
    config = GenerateContentConfig(
        system_instruction="You are a helpful assistant that can describe images."
    )

    if streaming:
        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=content,
            config=config,
        )
        for _ in response:
            ...
    else:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content,
            config=config,
        )
        assert response.text

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    # Verify core span attributes
    expected_attributes: Dict[str, Any] = {
        SpanAttributes.LLM_PROVIDER: "google",
        SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
        SpanAttributes.LLM_MODEL_NAME: "gemini-2.5-flash",
        SpanAttributes.INPUT_MIME_TYPE: "application/json",
        SpanAttributes.OUTPUT_MIME_TYPE: "application/json",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "system",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "You are a helpful assistant that can describe images.",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}": "text",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}": "Describe this image in one sentence.",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENTS}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}": "image",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "model",
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value, (
            f"Attribute '{key}' does not match. Expected: {expected_value!r}, Got: {attributes.get(key)!r}"
        )

    # Verify the image URL is the file URI passed in
    image_url_key = (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1."
        f"{MessageAttributes.MESSAGE_CONTENTS}.1."
        f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
    )
    assert attributes.get(image_url_key) == file_uri, (
        f"Expected image URL to be the file URI '{file_uri}', got: {attributes.get(image_url_key)!r}"
    )


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_validate_token_counts_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", "REDACTED"))
    response = client.models.generate_content_stream(
        model="gemini-2.5-flash",
        contents="What is the sum of the first 50 prime numbers? "
        "Generate and run code for the calculation, and make sure you get all 50.",
        config=GenerateContentConfig(tools=[Tool(code_execution=ToolCodeExecution)]),
    )
    chunks = []
    for chunk in response:
        chunks.append(chunk)
    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span, got {len(spans)}"
    span = spans[0]
    attributes = dict(span.attributes or {})

    assert chunks and hasattr(chunks[-1], "usage_metadata"), "Expected usage metadata"
    usage_metadata = chunks[-1].usage_metadata
    assert usage_metadata is not None, "Expected usage metadata to be present"
    prompt_token_count = 0
    if usage_metadata.prompt_token_count:
        prompt_token_count += usage_metadata.prompt_token_count
    if usage_metadata.tool_use_prompt_token_count:
        prompt_token_count += usage_metadata.tool_use_prompt_token_count
    completion_token_count = 0
    if usage_metadata.candidates_token_count:
        completion_token_count += usage_metadata.candidates_token_count
    if usage_metadata.thoughts_token_count:
        completion_token_count += usage_metadata.thoughts_token_count
    expected_attributes = {}
    if usage_metadata.total_token_count:
        expected_attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = usage_metadata.total_token_count
    if prompt_token_count:
        expected_attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = prompt_token_count
    if completion_token_count:
        expected_attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = completion_token_count
    if usage_metadata.thoughts_token_count:
        expected_attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING] = (
            usage_metadata.thoughts_token_count
        )
    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value, (
            f"Attribute {key} does not match expected value: got {attributes.get(key)}"
        )


def test_parse_args_handle_functools_wraps_outer_wrapper() -> None:
    import functools
    from inspect import signature

    from openinference.instrumentation.google_genai._wrappers import _parse_args

    class _Fake:
        def generate_content(self, *, model: str, contents: str) -> None:
            pass

    original = _Fake.generate_content

    @functools.wraps(original)
    def outer(self: Any, *args: Any, **kwargs: Any) -> None:
        return original(self, *args, **kwargs)

    instance = _Fake()
    request_parameters = _parse_args(
        signature(outer),
        instance,
        model="gemini-2.0-flash",
        contents="hello",
    )
    assert "self" not in request_parameters
    assert request_parameters["model"] == "gemini-2.0-flash"
    assert request_parameters["contents"] == "hello"
