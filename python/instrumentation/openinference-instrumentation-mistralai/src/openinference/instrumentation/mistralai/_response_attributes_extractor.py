import logging
from typing import (
    TYPE_CHECKING,
    Iterator,
    Tuple,
)

from openinference.semconv.trace import (
    SpanAttributes,
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
    # if usage := getattr(response, "usage", None):
    #     yield from _get_attributes_from_completion_usage(usage)
    # if (choices := getattr(response, "choices", None)) and isinstance(choices, Iterable):
    #     for choice in choices:
    #         if (index := getattr(choice, "index", None)) is None:
    #             continue
    #         if message := getattr(choice, "message", None):
    #             for key, value in _get_attributes_from_chat_completion_message(message):
    #                 yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value
