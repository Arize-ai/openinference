import logging
from enum import Enum
from types import ModuleType
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Tuple,
)

from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolCallAttributes,
)

__all__ = (
    "_RequestAttributesExtractor",
    "_get_attributes_from_ocr_process_param",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _RequestAttributesExtractor:
    __slots__ = (
        "_mistralai",
        "_chat_completion_type",
        "_completion_type",
        "_create_embedding_response_type",
    )

    def __init__(self, mistralai: ModuleType) -> None:
        self._mistralai = mistralai

    def get_attributes_from_request(
        self,
        request_parameters: Mapping[str, Any],
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not isinstance(request_parameters, Mapping):
            return
        yield from _get_attributes_from_chat_completion_create_param(request_parameters)

    def get_attributes_from_ocr_request(
        self, request_parameters: Mapping[str, Any]
    ) -> Iterator[Tuple[str, AttributeValue]]:
        if not isinstance(request_parameters, Mapping):
            return
        yield from _get_attributes_from_ocr_process_param(request_parameters)


def _get_attributes_from_chat_completion_create_param(
    params: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not isinstance(params, Mapping):
        return
    invocation_params = dict(params)
    invocation_params.pop("messages", None)
    invocation_params.pop("functions", None)
    invocation_params.pop("tools", None)
    yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)
    if (input_messages := params.get("messages")) and isinstance(input_messages, Iterable):
        # Use reversed() to get the last message first. This is because OTEL has a default limit of
        # 128 attributes per span, and flattening increases the number of attributes very quickly.
        for index, input_message in reversed(list(enumerate(input_messages))):
            for key, value in _get_attributes_from_message_param(input_message):
                yield f"{SpanAttributes.LLM_INPUT_MESSAGES}.{index}.{key}", value


def _is_base64_url(url: str) -> bool:
    """Check if a URL is a base64 data URL."""
    return url.startswith("data:") and "base64" in url


def _get_attributes_from_ocr_process_param(
    params: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not isinstance(params, Mapping):
        return

    # Extract model information
    model = params.get("model")
    if model:
        yield SpanAttributes.LLM_MODEL_NAME, model

    # Extract basic OCR parameters
    invocation_params = dict(params)
    # # Remove document from params as it might contain binary data
    # invocation_params.pop("document", None)
    yield SpanAttributes.LLM_INVOCATION_PARAMETERS, safe_json_dumps(invocation_params)

    # Extract document/image input as LLM input message (like OpenAI chat completions)
    document = params.get("document")
    if document:
        yield from _get_attributes_from_document_as_message(document)

    # Extract annotation format information
    bbox_format = params.get("bbox_annotation_format")
    if bbox_format:
        yield "ocr.bbox_annotation_format", safe_json_dumps(bbox_format)

    doc_format = params.get("document_annotation_format")
    if doc_format:
        yield "ocr.document_annotation_format", safe_json_dumps(doc_format)


def _get_attributes_from_document_as_message(
    document: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    """Convert document input to LLM input message format that Phoenix can display."""
    if not hasattr(document, "get"):
        return

    doc_type = document.get("type")

    # Create a synthetic LLM input message for the document
    # This follows the exact same pattern as OpenAI chat completions
    message_index = 0
    content_index = 0

    # Add role for the synthetic message
    yield (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_ROLE}",
        "user",
    )

    if doc_type == "image_url":
        # Handle image inputs - follow OpenAI pattern exactly
        if image_url := document.get("image_url"):
            yield (
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                "image",
            )
            yield (
                f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}",
                image_url,
            )

    elif doc_type == "document_url":
        # Handle document/PDF inputs
        if document_url := document.get("document_url"):
            # Determine if it's an image based on URL pattern
            if (
                _is_base64_url(document_url) and not document_url.startswith("data:application/pdf")
            ) or document_url.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                # Treat as image for Phoenix display
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "image",
                )
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}",
                    document_url,
                )
            else:
                # For PDFs and other documents, add as text content with URL reference
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
                    "text",
                )
                yield (
                    f"{SpanAttributes.LLM_INPUT_MESSAGES}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
                    f"Document: {document_url}",
                )


def _get_attributes_from_message_param(
    message: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not hasattr(message, "get"):
        return
    if role := message.get("role"):
        yield (
            MessageAttributes.MESSAGE_ROLE,
            role.value if isinstance(role, Enum) else role,
        )
    if content := message.get("content"):
        if isinstance(content, str):
            yield MessageAttributes.MESSAGE_CONTENT, content
        elif isinstance(content, List):
            try:
                json_string = safe_json_dumps(content)
            except Exception:
                logger.exception("Failed to serialize message content")
            else:
                yield MessageAttributes.MESSAGE_CONTENT, json_string
    if name := message.get("name"):
        yield MessageAttributes.MESSAGE_NAME, name
    if (function_call := message.get("function_call")) and hasattr(function_call, "get"):
        if function_name := function_call.get("name"):
            yield MessageAttributes.MESSAGE_FUNCTION_CALL_NAME, function_name
        if function_arguments := function_call.get("arguments"):
            yield (
                MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON,
                function_arguments,
            )
    if (tool_calls := message.get("tool_calls"),) and isinstance(tool_calls, Iterable):
        for index, tool_call in enumerate(tool_calls):
            if not hasattr(tool_call, "get"):
                continue
            if (function := tool_call.get("function")) and hasattr(function, "get"):
                if name := function.get("name"):
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}",
                        name,
                    )
                if arguments := function.get("arguments"):
                    yield (
                        f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}."
                        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        arguments,
                    )
