import logging
import warnings
from functools import lru_cache
from importlib.metadata import version
from types import ModuleType
from typing import (
    Any,
    Dict,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    cast,
)

from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes, AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.openai._with_span import _WithSpan
from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@lru_cache
def _get_openai_version() -> Tuple[int, int, int]:
    return cast(Tuple[int, int, int], tuple(map(int, version("openai").split(".")[:3])))


def _get_decision_type(openai: ModuleType) -> Optional[type]:
    """
    Return `openai.types.Decision`, the response type of the Decisions API, or `None` when the
    installed SDK predates it (the Decisions API was added in openai 3.26.0).
    """
    decision_type = getattr(getattr(openai, "types", None), "Decision", None)
    return decision_type if isinstance(decision_type, type) else None


class _ValueAndType(NamedTuple):
    value: str
    type: OpenInferenceMimeTypeValues


def _is_image_generation_call(item: Any) -> bool:
    if isinstance(item, Mapping):
        return item.get("type") == "image_generation_call"
    return getattr(item, "type", None) == "image_generation_call"


def _generated_image_results(obj: Any) -> Optional[Dict[str, Dict[int, Set[str]]]]:
    """
    Returns a `model_dump_json` exclude for the base64 `result` of the image_generation_call
    items in a Responses response, or `None` when it has none. Those images are recorded on
    `output.images.*`, where `base64_image_max_length` applies to them.
    """
    output = getattr(obj, "output", None)
    if not isinstance(output, list):
        return None
    indices = {i: {"result"} for i, item in enumerate(output) if _is_image_generation_call(item)}
    return {"output": indices} if indices else None


def _without_generated_image_results(request_parameters: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Returns the request parameters of a Responses request without the base64 `result` of the
    image_generation_call items that a later turn replays from an earlier turn's output, for
    `input.value`. The image was already recorded on the `output.images.*` of the span that
    generated it. The request parameters themselves are left as they are.
    """
    items = request_parameters.get("input")
    if not isinstance(items, list) or not any(map(_is_image_generation_call, items)):
        return request_parameters
    return {
        **request_parameters,
        "input": [
            {key: value for key, value in item.items() if key != "result"}
            if isinstance(item, Mapping) and _is_image_generation_call(item)
            else item
            for item in items
        ],
    }


def _io_value_and_type(obj: Any) -> _ValueAndType:
    if hasattr(obj, "model_dump_json") and callable(obj.model_dump_json):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # `warnings=False` in `model_dump_json()` is only supported in Pydantic v2
                value = obj.model_dump_json(
                    exclude_unset=True,
                    exclude=_generated_image_results(obj),
                )
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


class _HasAttributes(Protocol):
    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]: ...


def _finish_tracing(
    with_span: _WithSpan,
    has_attributes: _HasAttributes,
    status: Optional[trace_api.Status] = None,
) -> None:
    try:
        attributes: Attributes = dict(has_attributes.get_attributes())
    except Exception:
        logger.exception("Failed to get attributes")
        attributes = None
    try:
        extra_attributes: Attributes = dict(has_attributes.get_extra_attributes())
    except Exception:
        logger.exception("Failed to get extra attributes")
        extra_attributes = None
    try:
        with_span.finish_tracing(
            status=status,
            attributes=attributes,
            extra_attributes=extra_attributes,
        )
    except Exception:
        logger.exception("Failed to finish tracing")
