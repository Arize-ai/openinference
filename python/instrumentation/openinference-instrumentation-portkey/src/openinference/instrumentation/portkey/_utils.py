import logging
import warnings
from typing import Any, Iterable, Iterator, Mapping, NamedTuple, Optional, Sequence, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

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
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.INPUT_MIME_TYPE, value_and_type.type.value


def _as_output_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.OUTPUT_VALUE, value_and_type.value
    if value_and_type.type is not OpenInferenceMimeTypeValues.TEXT:
        yield SpanAttributes.OUTPUT_MIME_TYPE, value_and_type.type.value


def _finish_tracing(
    span: Optional[Span],
    attributes: Iterable[Tuple[str, AttributeValue]],
    extra_attributes: Iterable[Tuple[str, AttributeValue]],
    error: Optional[Exception] = None,
) -> None:
    if span is None:
        return

    try:
        attributes_dict = dict(attributes)
    except Exception:
        logger.exception("Failed to get attributes")
        attributes_dict = {}

    try:
        extra_attributes_dict = dict(extra_attributes)
    except Exception:
        logger.exception("Failed to get extra attributes")
        extra_attributes_dict = {}

    if error is not None:
        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
        span.record_exception(error)
        span.set_attribute(SpanAttributes.ERROR_TYPE, type(error).__name__)
        span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(error))
    else:
        span.set_status(trace_api.Status(trace_api.StatusCode.OK))

    for key, value in attributes_dict.items():
        span.set_attribute(key, value)

    for key, value in extra_attributes_dict.items():
        span.set_attribute(key, value)
