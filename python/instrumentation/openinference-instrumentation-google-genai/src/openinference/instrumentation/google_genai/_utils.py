import base64
import logging
import warnings
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.google_genai._with_span import _WithSpan
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceMimeTypeValues,
    SpanAttributes,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def get_attribute(obj: Any, attr_name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(attr_name, default)
    return getattr(obj, attr_name, default)


def _get_attributes_from_artifacts(
    inline_data: Any, tool_call_index: int
) -> Iterator[Tuple[str, AttributeValue]]:
    mime_type = get_attribute(inline_data, "mime_type")
    if (
        mime_type
        and "image" in mime_type
        and (data := get_attribute(inline_data, "data")) is not None
    ):
        prefix = f"{MessageAttributes.MESSAGE_CONTENTS}.{tool_call_index}."
        image_url = f"data:{inline_data.mime_type};base64,{base64.b64encode(data).decode()}"
        yield (
            f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}",
            image_url,
        )
        yield f"{prefix}{MessageContentAttributes.MESSAGE_CONTENT_TYPE}", "image"


def _get_attributes_from_content_text(
    text: str, index: int, only_text: bool
) -> Iterator[Tuple[str, AttributeValue]]:
    if only_text:
        yield MessageAttributes.MESSAGE_CONTENT, text
    else:
        yield (
            f"{MessageAttributes.MESSAGE_CONTENTS}.{index}.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}",
            text,
        )
        yield (
            f"{MessageAttributes.MESSAGE_CONTENTS}.{index}.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}",
            "text",
        )
