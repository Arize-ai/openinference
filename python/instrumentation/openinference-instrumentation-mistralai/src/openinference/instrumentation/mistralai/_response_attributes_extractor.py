import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
)

from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    SpanAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from mistralai.client.models import ChatCompletionResponse

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _get_attributes_from_chat_completion_response(response)


def _get_attributes_from_chat_completion_response(
    response: "ChatCompletionResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    if model := getattr(response, "model", None):
        yield SpanAttributes.LLM_MODEL_NAME, model
    yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.MISTRALAI.value
    yield SpanAttributes.LLM_SYSTEM, OpenInferenceLLMSystemValues.MISTRALAI.value
    if usage := getattr(response, "usage", None):
        yield from _get_attributes_from_completion_usage(usage)
    if (choices := getattr(response, "choices", None)) and isinstance(choices, Iterable):
        for choice in choices:
            if (index := _get_attribute_or_value(choice, "index")) is None:
                continue
            if message := _get_attribute_or_value(choice, "message"):
                for key, value in _get_attributes_from_chat_completion_message(message):
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value
            # Only capture finish_reason for the first choice.
            if index == 0:
                if (finish_reason := _get_attribute_or_value(choice, "finish_reason")) is not None:
                    yield SpanAttributes.LLM_FINISH_REASON, finish_reason


class _StreamResponseAttributesExtractor:
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _get_attributes_from_stream_chat_completion_response(response)


def _get_attributes_from_stream_chat_completion_response(
    response: Any,
) -> Iterator[Tuple[str, AttributeValue]]:
    data = response.data
    if model := data.get("model", None):
        yield SpanAttributes.LLM_MODEL_NAME, model
    yield SpanAttributes.LLM_PROVIDER, OpenInferenceLLMProviderValues.MISTRALAI.value
    yield SpanAttributes.LLM_SYSTEM, OpenInferenceLLMSystemValues.MISTRALAI.value
    if usage := data.get("usage", None):
        yield from _get_attributes_from_completion_usage(usage)
    if (choices := data.get("choices", None)) and isinstance(choices, Iterable):
        for choice in choices:
            if (index := _get_attribute_or_value(choice, "index")) is None:
                continue
            if message := _get_attribute_or_value(choice, "message"):
                for key, value in _get_attributes_from_chat_completion_message(message):
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value
            # Only capture finish_reason for the first choice.
            if index == 0:
                if (finish_reason := _get_attribute_or_value(choice, "finish_reason")) is not None:
                    yield SpanAttributes.LLM_FINISH_REASON, finish_reason


def _get_attributes_from_chat_completion_message(
    message: "ChatCompletionResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    if role := _get_attribute_or_value(message, "role"):
        yield MessageAttributes.MESSAGE_ROLE, role
    if content := _get_attribute_or_value(message, "content"):
        if isinstance(content, str):
            yield MessageAttributes.MESSAGE_CONTENT, content
        elif isinstance(content, Iterable):
            # Reasoning models answer with a list of chunks: a thinking chunk,
            # itself a list of text chunks, followed by the text of the answer.
            for index, chunk in enumerate(_flatten_content_chunks(content)):
                prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{index}"
                for key, value in chunk:
                    yield f"{prefix}.{key}", value
    if (tool_calls := _get_attribute_or_value(message, "tool_calls")) and isinstance(
        tool_calls, Iterable
    ):
        for index, tool_call in enumerate(tool_calls):
            if function := _get_attribute_or_value(tool_call, "function"):
                if name := _get_attribute_or_value(function, "name"):
                    yield (
                        (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
                        ),
                        name,
                    )
                if arguments := _get_attribute_or_value(function, "arguments"):
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
                    )


def _flatten_content_chunks(
    chunks: Iterable[Any],
) -> Iterator[List[Tuple[str, AttributeValue]]]:
    for chunk in chunks:
        chunk_type = _get_attribute_or_value(chunk, "type")
        if chunk_type == "thinking":
            for thought in _get_attribute_or_value(chunk, "thinking") or ():
                if text := _get_attribute_or_value(thought, "text"):
                    yield [
                        (MessageContentAttributes.MESSAGE_CONTENT_TYPE, "reasoning"),
                        (MessageContentAttributes.MESSAGE_CONTENT_TEXT, text),
                    ]
        elif chunk_type == "text":
            if text := _get_attribute_or_value(chunk, "text"):
                yield [
                    (MessageContentAttributes.MESSAGE_CONTENT_TYPE, "text"),
                    (MessageContentAttributes.MESSAGE_CONTENT_TEXT, text),
                ]


def _get_attributes_from_completion_usage(
    usage: object,
) -> Iterator[Tuple[str, AttributeValue]]:
    # openai.types.CompletionUsage
    # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/completion_usage.py#L8  # noqa: E501
    if (total_tokens := _get_attribute_or_value(usage, "total_tokens")) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens
    if (prompt_tokens := _get_attribute_or_value(usage, "prompt_tokens")) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (completion_tokens := _get_attribute_or_value(usage, "completion_tokens")) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens


def _get_attribute_or_value(
    obj: Any,
    attribute_name: str,
) -> Any:
    if (value := getattr(obj, attribute_name, None)) is not None or (
        hasattr(obj, "get") and callable(obj.get) and (value := obj.get(attribute_name)) is not None
    ):
        return value
    return None
