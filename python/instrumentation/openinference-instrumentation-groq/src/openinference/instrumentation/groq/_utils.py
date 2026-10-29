import logging
import warnings
from enum import Enum
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.groq._with_span import _WithSpan
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
    ToolCallAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    # Older SDKs keep fields they do not know about as plain dicts on the parent model,
    # newer ones type them, so both shapes have to be read the same way.
    if isinstance(obj, Mapping):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)


def _get_attributes_from_message_content(
    content: Mapping[str, Any],
) -> Iterator[Tuple[str, AttributeValue]]:
    type_ = content.get("type")
    if type_ == "text":
        yield MessageContentAttributes.MESSAGE_CONTENT_TYPE, "text"
        if text := content.get("text"):
            yield MessageContentAttributes.MESSAGE_CONTENT_TEXT, text
    elif type_ == "image_url":
        yield MessageContentAttributes.MESSAGE_CONTENT_TYPE, "image"
        if image := content.get("image_url"):
            if isinstance(image, str):
                if image:
                    yield (
                        f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
                        f"{ImageAttributes.IMAGE_URL}",
                        image,
                    )
            elif isinstance(image, Mapping):
                if url := image.get("url"):
                    yield (
                        f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}."
                        f"{ImageAttributes.IMAGE_URL}",
                        url,
                    )


def _get_attributes_from_message(message: Any) -> Iterator[Tuple[str, AttributeValue]]:
    """
    Flattens one chat message, either a request param (dict) or a response
    `ChatCompletionMessage`, into OpenInference message attributes (relative keys).
    """
    if role := get_attribute(message, "role"):
        yield MessageAttributes.MESSAGE_ROLE, role.value if isinstance(role, Enum) else role
    content = get_attribute(message, "content")
    # Reasoning models return their reasoning beside the answer when the request asks for
    # `reasoning_format="parsed"` (and accept it back on assistant turns). When present, the
    # message is emitted as ordered content blocks (reasoning, text, tool_use) the way the other
    # instrumentors do; otherwise the plain `message.content` form is kept.
    reasoning = get_attribute(message, "reasoning")
    content_index = 0
    if reasoning:
        prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}"
        yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "reasoning"
        yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", reasoning
        content_index += 1
        if content:
            prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}"
            yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "text"
            yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}", content
            content_index += 1
    elif content:
        if isinstance(content, str):
            yield MessageAttributes.MESSAGE_CONTENT, content
        elif isinstance(content, Sequence) and all(isinstance(part, Mapping) for part in content):
            # Multimodal requests send content as typed parts (text, image_url, ...).
            # Flatten each part into message contents so the attribute value stays a
            # primitive; a raw list of dicts is rejected by OpenTelemetry and dropped.
            # Parts keep their position in the original list, as in the openai
            # instrumentor, so an unsupported part leaves a gap in the indices.
            for part_index, part in enumerate(content):
                for key, value in _get_attributes_from_message_content(part):
                    yield (
                        f"{MessageAttributes.MESSAGE_CONTENTS}.{part_index}.{key}",
                        value,
                    )
        elif isinstance(content, Sequence):
            yield MessageAttributes.MESSAGE_CONTENT, safe_json_dumps(content)
    if name := get_attribute(message, "name"):
        yield MessageAttributes.MESSAGE_NAME, name
    if tool_call_id := get_attribute(message, "tool_call_id"):
        yield MessageAttributes.MESSAGE_TOOL_CALL_ID, tool_call_id
    # Deprecated by Groq
    if function_call := get_attribute(message, "function_call"):
        if function_name := get_attribute(function_call, "name"):
            yield MessageAttributes.MESSAGE_FUNCTION_CALL_NAME, function_name
        if function_arguments := get_attribute(function_call, "arguments"):
            yield MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON, function_arguments
    if (tool_calls := get_attribute(message, "tool_calls")) and isinstance(tool_calls, Iterable):
        for tool_call_index, tool_call in enumerate(tool_calls):
            tool_call_id = get_attribute(tool_call, "id")
            function = get_attribute(tool_call, "function")
            function_name = get_attribute(function, "name")
            function_arguments = get_attribute(function, "arguments")
            prefixes = [f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{tool_call_index}"]
            if reasoning:
                # Keep the tool call in the ordered view too, after the reasoning it came from.
                prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{content_index}"
                yield f"{prefix}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "tool_use"
                prefixes.append(prefix)
                content_index += 1
            for prefix in prefixes:
                if tool_call_id is not None:
                    yield f"{prefix}.{ToolCallAttributes.TOOL_CALL_ID}", tool_call_id
                if function_name:
                    yield f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}", function_name
                if function_arguments:
                    yield (
                        f"{prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}",
                        function_arguments,
                    )


class _ValueAndType(NamedTuple):
    value: str
    type: OpenInferenceMimeTypeValues


def _io_value_and_type(obj: Any) -> _ValueAndType:
    if hasattr(obj, "model_dump_json") and callable(obj.model_dump_json):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # `warnings=False` in `model_dump_json()` is only supported in Pydantic v2
                value = obj.model_dump_json(exclude_unset=True)
            assert isinstance(value, str)
        except Exception:
            logger.exception("Failed to get model dump json")
        else:
            return _ValueAndType(value, OpenInferenceMimeTypeValues.JSON)
    if not isinstance(obj, str) and isinstance(obj, (Sequence, Mapping)):
        try:
            value = safe_json_dumps(obj)
        except Exception:
            logger.exception("Failed to dump json")
        else:
            return _ValueAndType(value, OpenInferenceMimeTypeValues.JSON)
    return _ValueAndType(str(obj), OpenInferenceMimeTypeValues.TEXT)


def _as_input_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.INPUT_VALUE, value_and_type.value
    # It's assumed to be TEXT by default, so we can skip to save one attribute.
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.INPUT_MIME_TYPE, value_and_type.type.value


def _as_output_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.OUTPUT_VALUE, value_and_type.value
    # It's assumed to be TEXT by default, so we can skip to save one attribute.
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.OUTPUT_MIME_TYPE, value_and_type.type.value


def _finish_tracing(
    with_span: _WithSpan,
    attributes: Iterable[Tuple[str, AttributeValue]],
    extra_attributes: Iterable[Tuple[str, AttributeValue]],
    status: Optional[trace_api.Status] = None,
) -> None:
    try:
        attributes_dict = dict(attributes)
    except Exception:
        logger.exception("Failed to get attributes")
    try:
        extra_attributes_dict = dict(extra_attributes)
    except Exception:
        logger.exception("Failed to get extra attributes")
    try:
        with_span.finish_tracing(
            status=status,
            attributes=attributes_dict,
            extra_attributes=extra_attributes_dict,
        )
    except Exception:
        logger.exception("Failed to finish tracing")


def _materialize_content_iterables(kwargs: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Turns one-shot iterables (e.g. generators) in the chat messages into lists, so the
    message contents can be recorded on the span and still be sent by the SDK.
    """
    messages = kwargs.get("messages")
    if not isinstance(messages, Iterable) or isinstance(messages, (str, bytes, Mapping)):
        return kwargs
    changed = not isinstance(messages, Sequence)
    materialized = []
    for message in messages:
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, Iterable) and not isinstance(
                content, (str, bytes, Sequence, Mapping)
            ):
                message = {**message, "content": list(content)}
                changed = True
        materialized.append(message)
    if not changed:
        return kwargs
    return {**kwargs, "messages": materialized}
