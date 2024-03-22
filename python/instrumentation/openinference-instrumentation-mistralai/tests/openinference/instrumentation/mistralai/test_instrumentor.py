import json
from typing import (
    Any,
    Generator,
    Mapping,
    cast,
)

import pytest
import respx
from httpx import Response
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException
from mistralai.models.chat_completion import ChatMessage, FunctionCall, ToolCall, ToolChoice
from openinference.instrumentation.mistralai import MistralAIInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue


def test_synchronous_chat_completions_emits_expected_span(
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "a21b3c92f73642ccb6352ff9883c327b",
                "object": "chat.completion",
                "created": 1711044439,
                "model": "mistral-large-latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The 2018 FIFA World Cup was won by the French national team. They defeated Croatia 4-2 in the final, which took place on July 15, 2018, at the Luzhniki Stadium in Moscow, Russia. This was France's second World Cup title; they had previously won the tournament in 1998 when they hosted the event. Did you know that the 2018 World Cup was the first time the video assistant referee (VAR) system was used in a World Cup tournament? It played a significant role in several matches, helping referees make more accurate decisions by reviewing certain incidents.",  # noqa: E501
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {"prompt_tokens": 15, "total_tokens": 156, "completion_tokens": 141},
            },
        )
    )
    response = mistral_sync_client.chat(
        model="mistral-large-latest",
        messages=[
            ChatMessage(
                content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                role="user",
            )
        ],
        temperature=0.1,
    )
    choices = response.choices
    assert len(choices) == 1
    response_content = choices[0].message.content
    assert isinstance(response_content, str)
    assert "France" in response_content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok
    assert not span.status.description
    assert len(span.events) == 0

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "model": "mistral-large-latest",
        "temperature": 0.1,
    }

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2018? Answer in one word, no punctuation."
    )

    output_message_role = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}")
    assert output_message_role == "assistant"
    assert isinstance(
        output_message_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"),
        str,
    )
    assert "France" in output_message_content

    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 15
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 141
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 156
    assert attributes.pop(LLM_MODEL_NAME) == "mistral-large-latest"
    assert attributes == {}  # test should account for all span attributes


def test_synchronous_chat_completions_with_tool_call_response_emits_expected_spans(
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "6e8cf9abaac64c038c97e8db21e90567",
                "object": "chat.completion",
                "created": 1711062532,
                "model": "mistral-large-latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city": "San Francisco"}',
                                    }
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                        "logprobs": None,
                    }
                ],
                "usage": {"prompt_tokens": 96, "total_tokens": 119, "completion_tokens": 23},
            },
        )
    )
    response = mistral_sync_client.chat(
        model="mistral-large-latest",
        tool_choice=ToolChoice.any,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "finds the weather for a given city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city to find the weather for, e.g. 'London'",
                            }
                        },
                        "required": ["city"],
                    },
                },
            },
        ],
        messages=[
            ChatMessage(
                content="What's the weather like in San Francisco?",
                role="user",
            )
        ],
    )
    choices = response.choices
    assert len(choices) == 1
    assert choices[0].message.content == ""

    assert (tool_calls := choices[0].message.tool_calls)
    assert len(tool_calls) == 1
    assert (tool_call := tool_calls[0])
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city": "San Francisco"}'

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok
    assert not span.status.description
    assert len(span.events) == 0

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "model": "mistral-large-latest",
        "tool_choice": "any",
    }

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "What's the weather like in San Francisco?"
    )

    output_message_role = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}")
    assert output_message_role == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert isinstance(
        function_arguments_json := attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert json.loads(function_arguments_json) == {"city": "San Francisco"}

    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 96
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 23
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 119
    assert attributes.pop(LLM_MODEL_NAME) == "mistral-large-latest"
    assert attributes == {}  # test should account for all span attributes


def test_synchronous_chat_completions_with_tool_call_message_emits_expected_spans(
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "55ef30fc9c13499f92c77214ff056e7f",
                "object": "chat.completion",
                "created": 1711066504,
                "model": "mistral-large-latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The weather in San Francisco is currently sunny.",
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {"prompt_tokens": 64, "total_tokens": 74, "completion_tokens": 10},
            },
        )
    )
    response = mistral_sync_client.chat(
        model="mistral-large-latest",
        messages=[
            ChatMessage(
                content="What's the weather like in San Francisco?",
                role="user",
            ),
            ChatMessage(
                content="",
                role="assistant",
                tool_calls=[
                    ToolCall(
                        function=FunctionCall(
                            name="get_weather", arguments='{"city": "San Francisco"}'
                        )
                    )
                ],
            ),
            ChatMessage(role="tool", name="get_weather", content='{"weather_category": "sunny"}'),
        ],
    )
    choices = response.choices
    assert len(choices) == 1
    assert choices[0].message.content == "The weather in San Francisco is currently sunny."

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.status.is_ok
    assert not span.status.description
    assert len(span.events) == 0

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "model": "mistral-large-latest",
    }

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "What's the weather like in San Francisco?"
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert (
        attributes.pop(
            f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        )
        == '{"city": "San Francisco"}'
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "tool"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_NAME}") == "get_weather"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_CONTENT}")
        == '{"weather_category": "sunny"}'
    )

    output_message_role = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}")
    assert output_message_role == "assistant"
    assert isinstance(
        output_message_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"),
        str,
    )
    assert output_message_content == "The weather in San Francisco is currently sunny."

    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 64
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 10
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 74
    assert attributes.pop(LLM_MODEL_NAME) == "mistral-large-latest"
    assert attributes == {}  # test should account for all span attributes


def test_synchronous_chat_completions_emits_span_with_exception_event_on_error(
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            401,
            json={"message": "Unauthorized", "request_id": "2387442eca7ad4280697667d25a36f14"},
        )
    )
    with pytest.raises(MistralAPIException):
        mistral_sync_client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(
                    content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                    role="user",
                )
            ],
            temperature=0.1,
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert not span.status.is_ok
    assert isinstance(span.status.description, str)
    assert "Unauthorized" in span.status.description
    assert len(span.events) == 1
    event = span.events[0]
    assert event.name == "exception"

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "model": "mistral-large-latest",
        "temperature": 0.1,
    }

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2018? Answer in one word, no punctuation."
    )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.asyncio
async def test_asynchronous_chat_completions_emits_expected_span(
    mistral_async_client: MistralAsyncClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            200,
            json={
                "id": "a21b3c92f73642ccb6352ff9883c327b",
                "object": "chat.completion",
                "created": 1711044439,
                "model": "mistral-large-latest",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "The 2018 FIFA World Cup was won by the French national team. They defeated Croatia 4-2 in the final, which took place on July 15, 2018, at the Luzhniki Stadium in Moscow, Russia. This was France's second World Cup title; they had previously won the tournament in 1998 when they hosted the event. Did you know that the 2018 World Cup was the first time the video assistant referee (VAR) system was used in a World Cup tournament? It played a significant role in several matches, helping referees make more accurate decisions by reviewing certain incidents.",  # noqa: E501
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {"prompt_tokens": 15, "total_tokens": 156, "completion_tokens": 141},
            },
        )
    )
    response = await mistral_async_client.chat(
        model="mistral-large-latest",
        messages=[
            ChatMessage(
                content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                role="user",
            )
        ],
        temperature=0.1,
    )
    choices = response.choices
    assert len(choices) == 1
    response_content = choices[0].message.content
    assert isinstance(response_content, str)
    assert "France" in response_content

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok
    assert not span.status.description
    assert len(span.events) == 0

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "model": "mistral-large-latest",
        "temperature": 0.1,
    }

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2018? Answer in one word, no punctuation."
    )

    output_message_role = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}")
    assert output_message_role == "assistant"
    assert isinstance(
        output_message_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"),
        str,
    )
    assert "France" in output_message_content

    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 15
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 141
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 156
    assert attributes.pop(LLM_MODEL_NAME) == "mistral-large-latest"
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.asyncio
async def test_asynchronous_chat_completions_emits_span_with_exception_event_on_error(
    mistral_async_client: MistralAsyncClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            401,
            json={"message": "Unauthorized", "request_id": "2387442eca7ad4280697667d25a36f14"},
        )
    )
    with pytest.raises(MistralAPIException):
        await mistral_async_client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(
                    content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                    role="user",
                )
            ],
            temperature=0.1,
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert not span.status.is_ok
    assert isinstance(span.status.description, str)
    assert "Unauthorized" in span.status.description
    assert len(span.events) == 1
    event = span.events[0]
    assert event.name == "exception"

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert (
        OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
        == OpenInferenceMimeTypeValues.JSON
    )
    assert isinstance(invocation_parameters_str := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(invocation_parameters_str) == {
        "model": "mistral-large-latest",
        "temperature": 0.1,
    }

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "Who won the World Cup in 2018? Answer in one word, no punctuation."
    )
    assert attributes == {}  # test should account for all span attributes


@pytest.fixture(scope="module")
def mistral_sync_client() -> MistralClient:
    return MistralClient()


@pytest.fixture(scope="module")
def mistral_async_client() -> MistralAsyncClient:
    return MistralAsyncClient()


@pytest.fixture(scope="module")
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture(scope="module")
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> trace_api.TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    MistralAIInstrumentor().uninstrument()
    in_memory_span_exporter.clear()


OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
INPUT_VALUE = SpanAttributes.INPUT_VALUE
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_NAME = MessageAttributes.MESSAGE_NAME
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
