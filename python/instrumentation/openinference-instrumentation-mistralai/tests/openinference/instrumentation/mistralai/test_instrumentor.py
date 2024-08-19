import json
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Mapping,
    cast,
)

import pytest
import respx
from httpx import AsyncByteStream, Response
from mistralai.async_client import MistralAsyncClient
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException
from mistralai.models.chat_completion import (
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    FunctionCall,
    ToolCall,
    ToolChoice,
)
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.mistralai import MistralAIInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)


# Ensure we're using the common OITracer from common opeinference-instrumentation pkg
def test_oitracer() -> None:
    assert isinstance(MistralAIInstrumentor()._tracer, OITracer)


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_synchronous_chat_completions_emits_expected_span(
    use_context_attributes: bool,
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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

    def mistral_chat() -> ChatCompletionResponse:
        return mistral_sync_client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(
                    content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                    role="user",
                )
            ],
            temperature=0.1,
        )

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = mistral_chat()
    else:
        response = mistral_chat()
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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_synchronous_chat_completions_with_tool_call_response_emits_expected_spans(
    use_context_attributes: bool,
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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

    def mistral_chat() -> ChatCompletionResponse:
        tool = {
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
        }

        return mistral_sync_client.chat(
            model="mistral-large-latest",
            tool_choice=ToolChoice.any,
            tools=[tool],
            messages=[
                ChatMessage(
                    content="What's the weather like in San Francisco?",
                    role="user",
                )
            ],
        )

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = mistral_chat()
    else:
        response = mistral_chat()
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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_synchronous_chat_completions_with_tool_call_message_emits_expected_spans(
    use_context_attributes: bool,
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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

    def mistral_chat() -> ChatCompletionResponse:
        return mistral_sync_client.chat(
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
                ChatMessage(
                    role="tool", name="get_weather", content='{"weather_category": "sunny"}'
                ),
            ],
        )

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = mistral_chat()
    else:
        response = mistral_chat()
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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_synchronous_chat_completions_emits_span_with_exception_event_on_error(
    use_context_attributes: bool,
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            401,
            json={"message": "Unauthorized", "request_id": "2387442eca7ad4280697667d25a36f14"},
        )
    )

    def mistral_chat() -> ChatCompletionResponse:
        return mistral_sync_client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(
                    content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                    role="user",
                )
            ],
            temperature=0.1,
        )

    with pytest.raises(MistralAPIException):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                mistral_chat()
        else:
            mistral_chat()

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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_asynchronous_chat_completions_emits_expected_span(
    use_context_attributes: bool,
    mistral_async_client: MistralAsyncClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
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

    async def mistral_chat() -> ChatCompletionResponse:
        return await mistral_async_client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(
                    content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                    role="user",
                )
            ],
            temperature=0.1,
        )

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = await mistral_chat()
    else:
        response = await mistral_chat()
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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_asynchronous_chat_completions_emits_span_with_exception_event_on_error(
    use_context_attributes: bool,
    mistral_async_client: MistralAsyncClient,
    in_memory_span_exporter: InMemorySpanExporter,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            401,
            json={"message": "Unauthorized", "request_id": "2387442eca7ad4280697667d25a36f14"},
        )
    )

    async def mistral_chat() -> ChatCompletionResponse:
        return await mistral_async_client.chat(
            model="mistral-large-latest",
            messages=[
                ChatMessage(
                    content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                    role="user",
                )
            ],
            temperature=0.1,
        )

    with pytest.raises(MistralAPIException):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                await mistral_chat()
        else:
            await mistral_chat()

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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_synchronous_streaming_chat_completions_emits_expected_span(
    use_context_attributes: bool,
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    chat_stream: AsyncByteStream,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            200,
            stream=chat_stream,
        )
    )

    def mistral_stream() -> Iterable[ChatCompletionStreamResponse]:
        return mistral_sync_client.chat_stream(
            model="mistral-large-latest",
            messages=[
                ChatMessage(
                    content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                    role="user",
                )
            ],
            temperature=0.1,
        )

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response_stream = mistral_stream()
    else:
        response_stream = mistral_stream()
    response_content = ""
    for chunk in response_stream:
        if chunk_content := chunk.choices[0].delta.content:
            response_content += chunk_content

    assert (
        response_content
        == "The 2018 FIFA World Cup was won by the French national team. They defeated Croatia 4-2 in the final, which took place on July 15, 2018, in Moscow, Russia. This was France's second World Cup title; they had previously won the tournament in 1998 when they hosted the event. Did you know that the World Cup is the most prestigious tournament in international football and is often considered as the height of a footballer's career?"  # noqa: E501
    )  # noqa: E501

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok
    assert not span.status.description
    assert len(span.events) == 1
    event = span.events[0]
    assert event.name == "First Token Stream Event"

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
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 109
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 124
    assert attributes.pop(LLM_MODEL_NAME) == "mistral-large-latest"
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.asyncio
@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_asynchronous_streaming_chat_completions_emits_expected_span(
    use_context_attributes: bool,
    mistral_async_client: MistralAsyncClient,
    in_memory_span_exporter: InMemorySpanExporter,
    chat_stream: AsyncByteStream,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            200,
            stream=chat_stream,
        )
    )

    async def mistral_stream() -> AsyncIterator[ChatCompletionStreamResponse]:
        return mistral_async_client.chat_stream(
            model="mistral-large-latest",
            messages=[
                ChatMessage(
                    content="Who won the World Cup in 2018? Answer in one word, no punctuation.",
                    role="user",
                )
            ],
            temperature=0.1,
        )

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response_stream = await mistral_stream()
    else:
        response_stream = await mistral_stream()
    response_content = ""
    async for chunk in response_stream:
        if chunk_content := chunk.choices[0].delta.content:
            response_content += chunk_content

    assert (
        response_content
        == "The 2018 FIFA World Cup was won by the French national team. They defeated Croatia 4-2 in the final, which took place on July 15, 2018, in Moscow, Russia. This was France's second World Cup title; they had previously won the tournament in 1998 when they hosted the event. Did you know that the World Cup is the most prestigious tournament in international football and is often considered as the height of a footballer's career?"  # noqa: E501
    )  # noqa: E501

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok
    assert not span.status.description
    assert len(span.events) == 1
    event = span.events[0]
    assert event.name == "First Token Stream Event"

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
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 109
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 124
    assert attributes.pop(LLM_MODEL_NAME) == "mistral-large-latest"
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_synchronous_streaming_chat_completions_with_tool_call_response_emits_expected_spans(
    use_context_attributes: bool,
    mistral_sync_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter,
    chat_stream_with_tool_call: AsyncByteStream,
    respx_mock: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(
            200,
            stream=chat_stream_with_tool_call,
        )
    )
    tool = {
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
    }

    def mistral_chat() -> Iterable[ChatCompletionStreamResponse]:
        return mistral_sync_client.chat_stream(
            model="mistral-large-latest",
            tool_choice=ToolChoice.any,
            tools=[tool],
            messages=[
                ChatMessage(
                    content="What's the weather like in San Francisco?",
                    role="user",
                )
            ],
        )

    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response_stream = mistral_chat()
    else:
        response_stream = mistral_chat()

    for chunk in response_stream:
        delta = chunk.choices[0].delta
        assert not delta.content
        if tool_calls := delta.tool_calls:
            tool_call = tool_calls[0]
            assert tool_call.function.name == "get_weather"
            assert tool_call.function.arguments == '{"city": "San Francisco"}'

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.status.is_ok
    assert not span.status.description
    assert len(span.events) == 1
    event = span.events[0]
    assert event.name == "First Token Stream Event"

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
    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )
    assert attributes == {}  # test should account for all span attributes


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    assert attributes.pop(SESSION_ID, None) == session_id
    assert attributes.pop(USER_ID, None) == user_id
    attr_metadata = attributes.pop(METADATA, None)
    assert attr_metadata is not None
    assert isinstance(attr_metadata, str)  # must be json string
    metadata_dict = json.loads(attr_metadata)
    assert metadata_dict == metadata
    attr_tags = attributes.pop(TAG_TAGS, None)
    assert attr_tags is not None
    assert len(attr_tags) == len(tags)
    assert list(attr_tags) == tags
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE, None) == prompt_template
    assert (
        attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, None) == prompt_template_version
    )
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None) == json.dumps(
        prompt_template_variables
    )


@pytest.fixture()
def session_id() -> str:
    return "my-test-session-id"


@pytest.fixture()
def user_id() -> str:
    return "my-test-user-id"


@pytest.fixture()
def metadata() -> Dict[str, Any]:
    return {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }


@pytest.fixture()
def tags() -> List[str]:
    return ["tag-1", "tag-2"]


@pytest.fixture
def prompt_template() -> str:
    return (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )


@pytest.fixture
def prompt_template_version() -> str:
    return "v1.0"


@pytest.fixture
def prompt_template_variables() -> Dict[str, Any]:
    return {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }


@pytest.fixture(scope="module")
def mistral_sync_client() -> MistralClient:
    return MistralClient(api_key="123")


@pytest.fixture(scope="module")
def mistral_async_client() -> MistralAsyncClient:
    return MistralAsyncClient(api_key="123")


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


@pytest.fixture
def server_sent_events() -> List[bytes]:
    return [
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"The"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" "},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"2"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"0"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"8"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" FIFA"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" Cup"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" was"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" won"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" by"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" French"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" national"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" team"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"."},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" They"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" defeated"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" Cro"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"at"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"ia"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" "},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"4"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"-"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"2"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" in"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" final"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":","},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" which"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" took"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" place"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" on"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" July"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" "},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"5"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":","},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" "},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"2"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"0"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"8"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":","},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" in"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" Moscow"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":","},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" Russia"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"."},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" This"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" was"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" France"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"'"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"s"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" second"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" Cup"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" title"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":";"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" they"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" had"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" previously"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" won"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" tournament"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" in"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" "},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"9"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"9"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"8"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" when"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" they"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" hosted"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" event"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"."},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" Did"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" you"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" know"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" that"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" World"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" Cup"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" most"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" prest"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"igious"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" tournament"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" in"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" international"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" football"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" and"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" often"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" considered"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" as"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" the"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" height"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" of"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" a"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" football"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"er"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"'"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"s"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":" career"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":"?"},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"ca75aff0161b45248d217b410da72ff3","object":"chat.completion.chunk","created":1711130222,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":15,"total_tokens":124,"completion_tokens":109}}\n\n""",  # noqa: E501
        b"""data: [DONE]\n""",
    ]


@pytest.fixture
def server_sent_events_with_tool_call() -> List[bytes]:
    return [
        b"""data: {"id":"2b081bc20de346a987269f670396c651","object":"chat.completion.chunk","created":1711157727,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null,"logprobs":null}]}\n\n""",  # noqa: E501
        b"""data: {"id":"7a5fe619e34d4d0fb02dedcbdd17b8b4","object":"chat.completion.chunk","created":1711159624,"model":"mistral-large-latest","choices":[{"index":0,"delta":{"content":null,"tool_calls":[{"function":{"name":"get_weather","arguments":"{\\"city\\": \\"San Francisco\\"}"}}]},"finish_reason":"tool_calls","logprobs":null}],"usage":{"prompt_tokens":96,"total_tokens":119,"completion_tokens":23}}\n\n""",  # noqa: E501
        b"""data: [DONE]\n""",
    ]


class MockAsyncByteStream(AsyncByteStream):
    def __init__(self, byte_stream: Iterable[bytes]):
        self._byte_stream = byte_stream

    def __iter__(self) -> Iterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for byte_string in self._byte_stream:
            yield byte_string


@pytest.fixture
def chat_stream(server_sent_events: List[bytes]) -> AsyncByteStream:
    return MockAsyncByteStream(server_sent_events)


@pytest.fixture
def chat_stream_with_tool_call(server_sent_events_with_tool_call: List[bytes]) -> AsyncByteStream:
    return MockAsyncByteStream(server_sent_events_with_tool_call)


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
SESSION_ID = SpanAttributes.SESSION_ID
USER_ID = SpanAttributes.USER_ID
METADATA = SpanAttributes.METADATA
TAG_TAGS = SpanAttributes.TAG_TAGS
