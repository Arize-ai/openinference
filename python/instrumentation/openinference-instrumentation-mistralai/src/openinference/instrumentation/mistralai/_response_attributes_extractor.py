import logging
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Tuple,
)

from openinference.semconv.trace import (
    MessageAttributes,
    SpanAttributes,
    ToolCallAttributes,
)
from opentelemetry.util.types import AttributeValue

if TYPE_CHECKING:
    from mistralai.models.chat_completion import ChatCompletionResponse

__all__ = ("_ResponseAttributesExtractor",)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _ResponseAttributesExtractor:
    def get_attributes_from_response(
        self,
        response: "ChatCompletionResponse",
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _get_attributes_from_chat_completion_response(response)


def _get_attributes_from_chat_completion_response(
    response: "ChatCompletionResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/chat/chat_completion.py#L40  # noqa: E501
    if model := getattr(response, "model", None):
        yield SpanAttributes.LLM_MODEL_NAME, model
    if usage := getattr(response, "usage", None):
        yield from _get_attributes_from_completion_usage(usage)
    if (choices := getattr(response, "choices", None)) and isinstance(choices, Iterable):
        for choice in choices:
            if (index := getattr(choice, "index", None)) is None:
                continue
            if message := getattr(choice, "message", None):
                for key, value in _get_attributes_from_chat_completion_message(message):
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value


def _get_attributes_from_chat_completion_message(
    message: "ChatCompletionResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    if role := getattr(message, "role", None):
        yield MessageAttributes.MESSAGE_ROLE, role
    if content := getattr(message, "content", None):
        yield MessageAttributes.MESSAGE_CONTENT, content
    if (tool_calls := getattr(message, "tool_calls", None)) and isinstance(tool_calls, Iterable):
        for index, tool_call in enumerate(tool_calls):
            if function := getattr(tool_call, "function", None):
                if name := getattr(function, "name", None):
                    yield (
                        (
                            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
                        ),
                        name,
                    )
                if arguments := getattr(function, "arguments", None):
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
                    )


def _get_attributes_from_completion_usage(
    usage: object,
) -> Iterator[Tuple[str, AttributeValue]]:
    # openai.types.CompletionUsage
    # See https://github.com/openai/openai-python/blob/f1c7d714914e3321ca2e72839fe2d132a8646e7f/src/openai/types/completion_usage.py#L8  # noqa: E501
    if (total_tokens := getattr(usage, "total_tokens", None)) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_TOTAL, total_tokens
    if (prompt_tokens := getattr(usage, "prompt_tokens", None)) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_PROMPT, prompt_tokens
    if (completion_tokens := getattr(usage, "completion_tokens", None)) is not None:
        yield SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, completion_tokens
