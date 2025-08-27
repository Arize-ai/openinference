# type: ignore
# ruff: noqa: E501
import json
from typing import Any, Dict, Iterator

import pytest
from google import genai
from google.genai.types import Content, FunctionDeclaration, GenerateContentConfig, Part, Tool
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from openinference.semconv.trace import (
    MessageAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


class Answer(BaseModel):
    answer: str


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
    api_key = "REDACTED"

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
                "top_k": 40,
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
    api_key = "REDACTED"

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
        or "",
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
    api_key = "REDACTED"

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
        or "",
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
    api_key = "REDACTED"

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
def test_chat_session_with_tool(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
    setup_google_genai_instrumentation: None,
) -> None:
    # REDACT API Key, Cassette has stored response, delete cassette and replace API Key to edit test
    api_key = "REDACTED"

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

    # Extract attributes using the response extractor
    extractor = _ResponseExtractor(accumulator)
    attributes = dict(extractor.get_extra_attributes())

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

    # Verify token counts from final chunk
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 60
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 50
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 10


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
    api_key = "REDACTED"

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
