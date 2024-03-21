import json
import os
from typing import (
    Generator,
    Mapping,
    cast,
)

import pytest
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openinference.instrumentation.mistralai import MistralAIInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
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


def test_synchronous_chat_completions(
    mistral_client: MistralClient,
    in_memory_span_exporter: InMemorySpanExporter
) -> None:
    response = mistral_client.chat(
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
    # assert span.status.is_ok
    # assert not span.status.description
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    # assert isinstance(attributes.pop(INPUT_VALUE), str)
    # assert (
    #     OpenInferenceMimeTypeValues(attributes.pop(INPUT_MIME_TYPE))
    #     == OpenInferenceMimeTypeValues.JSON
    # )
    # assert (
    #     json.loads(cast(str, attributes.pop(LLM_INVOCATION_PARAMETERS)))
    #     == {"model": "mistral-large-latest", "temperature": 0.1}
    # )

    # input_messages = attributes.pop(LLM_INPUT_MESSAGES)
    # assert isinstance(input_messages, list)
    # assert len(input_messages) == 1
    # input_message = input_messages[0]
    # assert input_message == {
    #     "role": "user",
    #     "content": "Who won the World Cup in 2018? Answer in one word, no punctuation.",
    # }

    # output_messages = attributes.pop(LLM_OUTPUT_MESSAGES)
    # assert isinstance(output_messages, list)
    # assert len(output_messages) == 1
    # output_message = output_messages[0]
    # assert output_message == {
    #     "role": "assistant",
    #     "content": "France won the World Cup in 2018.",
    # }

    # assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    # assert (
    #     OpenInferenceMimeTypeValues(attributes.pop(OUTPUT_MIME_TYPE))
    #     == OpenInferenceMimeTypeValues.JSON
    # )
    # assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 17
    # assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 18
    # assert (
    #     attributes.pop(LLM_TOKEN_COUNT_COMPLETION)
    #     == 19
    # )
    # assert attributes.pop(LLM_MODEL_NAME) == "mistral-large-latest"
    # assert attributes == {}  # test should account for all span attributes


@pytest.fixture(scope="module")
def mistral_client() -> MistralClient:
    return MistralClient()


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
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
