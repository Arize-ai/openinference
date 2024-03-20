import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from openinference.semconv.trace import (
    EmbeddingAttributes,
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)


def test_synchronous_chat_completions(
    # respx_mock: MockRouter,
    # in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    client = MistralClient(api_key=os.environ["MISTRAL_API_KEY"])
    response = client.chat(
        model="mistral-large-latest",
        messages=[
            ChatMessage(
                content="Who won the World Cup in 2018? Answer in one word.",
                role="user",
            )
        ],
    )
    choices = response.choices
    assert len(choices) == 1
    response_content = choices[0].message.content
    assert isinstance(response_content, str)
    assert "france" in response_content.lower()


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
