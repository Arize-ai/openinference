import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Iterator,
    Mapping,
    Tuple,
)

from opentelemetry.util.types import AttributeValue

from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

if TYPE_CHECKING:
    from mistralai.models import ChatCompletionResponse
    from mistralai.models.ocrresponse import OCRResponse

__all__ = (
    "_ResponseAttributesExtractor",
    "_OCRResponseAttributesExtractor",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _OCRResponseAttributesExtractor:
    def get_attributes_from_response(
        self,
        response: Any,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        yield from _get_attributes_from_ocr_response(response)


def _get_attributes_from_ocr_usage(
    usage_info: object,
) -> Iterator[Tuple[str, AttributeValue]]:
    """Extract usage information from OCR usage info."""
    if (pages_processed := _get_attribute_or_value(usage_info, "pages_processed")) is not None:
        yield "ocr.pages_processed", pages_processed

    if (doc_size_bytes := _get_attribute_or_value(usage_info, "doc_size_bytes")) is not None:
        yield "ocr.document_size_bytes", doc_size_bytes


def _get_attributes_from_ocr_response(
    response: "OCRResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    # Extract model name
    if model := getattr(response, "model", None):
        yield SpanAttributes.LLM_MODEL_NAME, model

    # Extract usage information
    if usage_info := getattr(response, "usage_info", None):
        yield from _get_attributes_from_ocr_usage(usage_info)

    # Extract document annotation if present - this is the main output
    if document_annotation := getattr(response, "document_annotation", None):
        yield SpanAttributes.OUTPUT_VALUE, document_annotation

    # Structure OCR output as LLM output messages - one message per page for Phoenix display
    if (pages := getattr(response, "pages", None)) and isinstance(pages, Iterable):
        for page_index, page in enumerate(pages):
            message_index = page_index  # Each page gets its own message
            content_index = 0

            # Add role for this page's output message
            yield (
                f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_ROLE}",
                "assistant",
            )

            # Add markdown content for this page
            if markdown := _get_attribute_or_value(page, "markdown"):
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "text",
                )
                yield (
                    f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                    markdown,
                )
                content_index += 1

            # Add extracted images from this page - keep it simple and robust
            if (images := _get_attribute_or_value(page, "images")) and isinstance(images, Iterable):
                for image in images:
                    if image_base64 := _get_attribute_or_value(image, "image_base64"):
                        # Add image content
                        yield (
                            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                            "image",
                        )
                        yield (
                            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}",
                            image_base64,
                        )
                        content_index += 1

        # Keep basic structured data for retrieval context
        for page_index, page in enumerate(pages):
            if markdown := _get_attribute_or_value(page, "markdown"):
                yield f"retrieval.documents.{page_index}.document.content", markdown
                yield (
                    f"retrieval.documents.{page_index}.document.metadata",
                    f'{{"type": "ocr_page", "page_index": {page_index}}}',
                )


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
    if usage := getattr(response, "usage", None):
        yield from _get_attributes_from_completion_usage(usage)
    if (choices := getattr(response, "choices", None)) and isinstance(choices, Iterable):
        for choice in choices:
            if (index := _get_attribute_or_value(choice, "index")) is None:
                continue
            if message := _get_attribute_or_value(choice, "message"):
                for key, value in _get_attributes_from_chat_completion_message(message):
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value


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
    if usage := data.get("usage", None):
        yield from _get_attributes_from_completion_usage(usage)
    if (choices := data.get("choices", None)) and isinstance(choices, Iterable):
        for choice in choices:
            if (index := _get_attribute_or_value(choice, "index")) is None:
                continue
            if message := _get_attribute_or_value(choice, "message"):
                for key, value in _get_attributes_from_chat_completion_message(message):
                    yield f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{index}.{key}", value


def _get_attributes_from_chat_completion_message(
    message: "ChatCompletionResponse",
) -> Iterator[Tuple[str, AttributeValue]]:
    if role := _get_attribute_or_value(message, "role"):
        yield MessageAttributes.MESSAGE_ROLE, role
    if content := _get_attribute_or_value(message, "content"):
        yield MessageAttributes.MESSAGE_CONTENT, content
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
