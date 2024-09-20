import json
from typing import Any, Dict, Generator, Optional

import anthropic
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.resources.completions import AsyncCompletions, Completions
from anthropic.resources.messages import (
    AsyncMessages,
    Messages,
)
from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from typing_extensions import assert_never
from wrapt import BoundFunctionWrapper

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)


def remove_all_vcr_request_headers(request: Any) -> Any:
    """
    Removes all request headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_request_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes all response headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    response["headers"] = {}
    return response


def _to_assistant_message_param(
    message: Message,
) -> MessageParam:
    content = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append(block)
        elif isinstance(block, ToolUseBlock):
            content.append(block)  # type: ignore
        else:
            assert_never(block)
    return MessageParam(content=content, role="assistant")


def _get_tool_use_id(message: Message) -> Optional[str]:
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            return block.id
    return None


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_anthropic_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AnthropicInstrumentor().uninstrument()


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_completions_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="fake")

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" why is the sky blue? respond in five words or less."
        f" {anthropic.AI_PROMPT}"
    )

    stream = client.completions.create(
        model="claude-2.1",
        prompt=prompt,
        max_tokens_to_sample=1000,
        stream=True,
    )
    for event in stream:
        print(event.completion)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Completions"
    attributes = dict(spans[0].attributes or {})
    print(attributes)

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)

    invocation_params = {"model": "claude-2.1", "max_tokens_to_sample": 1000, "stream": True}
    assert json.loads(inv_params) == invocation_params
    assert attributes.pop(LLM_OUTPUT_MESSAGES) == " Light scatters blue."


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_anthropic_instrumentation_async_completions_streaming(
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="fake")

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" why is the sky blue? respond in five words or less."
        f" {anthropic.AI_PROMPT}"
    )

    stream = await client.completions.create(
        model="claude-2.1",
        prompt=prompt,
        max_tokens_to_sample=1000,
        stream=True,
    )
    async for event in stream:
        print(event.completion)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncCompletions"
    attributes = dict(spans[0].attributes or {})
    print(attributes)

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)

    invocation_params = {"model": "claude-2.1", "max_tokens_to_sample": 1000, "stream": True}
    assert json.loads(inv_params) == invocation_params
    assert attributes.pop(LLM_OUTPUT_MESSAGES) == " Light scatters blue."


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_completions(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="fake")

    invocation_params = {"model": "claude-2.1", "max_tokens_to_sample": 1000}

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" how does a court case get to the Supreme Court?"
        f" {anthropic.AI_PROMPT}"
    )

    client.completions.create(
        model="claude-2.1",
        prompt=prompt,
        max_tokens_to_sample=1000,
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Completions"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_messages(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="fake")
    input_message = "What's the capital of France?"

    invocation_params = {"max_tokens": 1024, "model": "claude-3-opus-20240229"}

    client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-3-opus-20240229",
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Messages"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        msg_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "paris" in msg_content.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-3-opus-20240229"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_messages_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="fake")
    input_message = "Why is the sky blue? Answer in 5 words or less"

    invocation_params = {"max_tokens": 1024, "model": "claude-2.1", "stream": True}

    stream = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-2.1",
        stream=True,
    )

    for event in stream:
        print(event)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Messages"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        msg_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "Light scatters blue." in msg_content
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 21
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 10
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 31

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    # TODO(harrison): the output here doesn't look properly
    # serialized but looks like openai, mistral accumulators do
    # the same thing. need to look into why this might be wrong
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_anthropic_instrumentation_async_messages_streaming(
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="fake")
    input_message = "Why is the sky blue? Answer in 5 words or less"

    invocation_params = {"max_tokens": 1024, "model": "claude-2.1", "stream": True}

    stream = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-2.1",
        stream=True,
    )

    async for event in stream:
        print(event)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncMessages"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        msg_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "Light scatters blue." in msg_content
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 21
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 10
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 31

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    # TODO(harrison): the output here doesn't look properly
    # serialized but looks like openai, mistral accumulators do
    # the same thing. need to look into why this might be wrong
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes

@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_anthropic_instrumentation_async_completions(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="fake")

    invocation_params = {"model": "claude-2.1", "max_tokens_to_sample": 1000}

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" how does a court case get to the Supreme Court?"
        f" {anthropic.AI_PROMPT}"
    )

    await client.completions.create(
        model="claude-2.1",
        prompt=prompt,
        max_tokens_to_sample=1000,
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncCompletions"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_anthropic_instrumentation_async_messages(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="fake")
    input_message = "What's the capital of France?"

    invocation_params = {"max_tokens": 1024, "model": "claude-3-opus-20240229"}

    await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-3-opus-20240229",
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "AsyncMessages"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(
        msg_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "paris" in msg_content.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-3-opus-20240229"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_multiple_tool_calling(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = anthropic.Anthropic(api_key="fake")

    input_message = (
        "What is the weather like right now in New York?"
        " Also what time is it there? Use necessary tools simultaneously."
    )

    client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature,"
                            " either 'celsius' or 'fahrenheit'",
                        },
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_time",
                "description": "Get the current time in a given time zone",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The IANA time zone name, e.g. America/Los_Angeles",
                        }
                    },
                    "required": ["timezone"],
                },
            },
        ],
        messages=[{"role": "user", "content": input_message}],
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Messages"
    attributes = dict(spans[0].attributes or {})

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert isinstance(
        attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "get_time"
    )
    assert isinstance(
        attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert isinstance(attributes.pop(OUTPUT_MIME_TYPE), str)
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_multiple_tool_calling_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = anthropic.Anthropic(api_key="fake")

    input_message = (
        "What is the weather like right now in New York?"
        " Also what time is it there? Use necessary tools simultaneously."
    )

    stream = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature,"
                            " either 'celsius' or 'fahrenheit'",
                        },
                    },
                    "required": ["location"],
                },
            },
            {
                "name": "get_time",
                "description": "Get the current time in a given time zone",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The IANA time zone name, e.g. America/Los_Angeles",
                        }
                    },
                    "required": ["timezone"],
                },
            },
        ],
        messages=[{"role": "user", "content": input_message}],
        stream=True,
    )
    for event in stream:
        print(event)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "Messages"
    attributes = dict(spans[0].attributes or {})

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "get_time"
    )
    get_time_input_str = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    json.loads(get_time_input_str) == {"timezone": "America/New_York"}  # type: ignore
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    get_weather_input_str = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    assert json.loads(get_weather_input_str) == {"location": "New York, NY", "unit": "celsius"}  # type: ignore
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 518
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 149
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 667
    # TODO(harrison): the output here doesn't look properly
    # serialized but looks like openai, mistral accumulators do
    # the same thing. need to look into why this might be wrong
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
@pytest.mark.parametrize(
    "assistant_message",
    (
        pytest.param(
            {
                "content": [
                    TextBlock(
                        text="Certainly! I can help you get the current weather information for"
                        " San Francisco in Fahrenheit. To do this, I'll use the get_weather"
                        " function. Let me fetch that information for you right away.",
                        type="text",
                    ),
                    ToolUseBlock(
                        id="toolu_01KBqpqR73qWGsMaW3vBzEjz",
                        input={"location": "San Francisco, CA", "unit": "fahrenheit"},
                        name="get_weather",
                        type="tool_use",
                    ),
                ],
                "role": "assistant",
            },
            id="with_blocks",
        ),
        pytest.param(
            {
                "content": [
                    TextBlockParam(
                        text="Certainly! I can help you get the current weather information for"
                        " San Francisco in Fahrenheit. To do this, I'll use the get_weather"
                        " function. Let me fetch that information for you right away.",
                        type="text",
                    ),
                    ToolUseBlockParam(
                        id="toolu_01KBqpqR73qWGsMaW3vBzEjz",
                        input={"location": "San Francisco, CA", "unit": "fahrenheit"},
                        name="get_weather",
                        type="tool_use",
                    ),
                ],
                "role": "assistant",
            },
            id="with_block_params",
        ),
    ),
)
def test_anthropic_instrumentation_tool_use_in_input(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    assistant_message: MessageParam,
) -> None:
    client = anthropic.Anthropic(api_key="fake")
    messages = [
        {"role": "user", "content": "What is the weather like in San Francisco in Fahrenheit?"},
        assistant_message,
        MessageParam(
            content=[
                ToolResultBlockParam(
                    tool_use_id="toolu_01KBqpqR73qWGsMaW3vBzEjz",
                    content='{"weather": "sunny", "temperature": "75"}',
                    type="tool_result",
                    is_error=False,
                )
            ],
            role="user",
        ),
    ]

    client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature,"
                            ' either "celsius" or "fahrenheit"',
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        messages=messages,  # type: ignore
    )

    spans = in_memory_span_exporter.get_finished_spans()

    attributes = dict(spans[0].attributes or {})

    assert (
        attributes.get(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert (
        attributes.get(
            f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        )
        == '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
    )
    assert attributes.get(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"

    assert (
        attributes.get(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_CONTENT}")
        == '{"weather": "sunny", "temperature": "75"}'
    )
    assert attributes.get(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "user"


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_context_attributes_existence(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    session_id = "my-test-session-id"
    user_id = "my-test-user-id"
    metadata = {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }
    tags = ["tag-1", "tag-2"]
    prompt_template = (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )
    prompt_template_version = "v1.0"
    prompt_template_variables = {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }

    client = Anthropic(api_key="fake")

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" how does a court case get to the Supreme Court?"
        f" {anthropic.AI_PROMPT}"
    )

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        client.completions.create(
            model="claude-2.1",
            prompt=prompt,
            max_tokens_to_sample=1000,
        )

    spans = in_memory_span_exporter.get_finished_spans()

    for span in spans:
        att = dict(span.attributes or {})
        assert att.get(SESSION_ID, None)
        assert att.get(USER_ID, None)
        assert att.get(METADATA, None)
        assert att.get(TAG_TAGS, None)
        assert att.get(LLM_PROMPT_TEMPLATE, None)
        assert att.get(LLM_PROMPT_TEMPLATE_VERSION, None)
        assert att.get(LLM_PROMPT_TEMPLATE_VARIABLES, None)


def test_anthropic_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

    assert isinstance(Completions.create, BoundFunctionWrapper)
    assert isinstance(Messages.create, BoundFunctionWrapper)
    assert isinstance(AsyncCompletions.create, BoundFunctionWrapper)
    assert isinstance(AsyncMessages.create, BoundFunctionWrapper)

    AnthropicInstrumentor().uninstrument()

    assert not isinstance(Completions.create, BoundFunctionWrapper)
    assert not isinstance(Messages.create, BoundFunctionWrapper)
    assert not isinstance(AsyncCompletions.create, BoundFunctionWrapper)
    assert not isinstance(AsyncMessages.create, BoundFunctionWrapper)


# Ensure we're using the common OITracer from common openinference-instrumentation pkg
def test_oitracer(
    setup_anthropic_instrumentation: Any,
) -> None:
    assert isinstance(AnthropicInstrumentor()._tracer, OITracer)


CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
USER_ID = SpanAttributes.USER_ID
