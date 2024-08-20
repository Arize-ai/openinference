import logging
import warnings
from functools import lru_cache
from importlib.metadata import version
from typing import (
    Any,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
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


def _get_texts(
    model_input: Optional[Union[str, List[str], List[int], List[List[int]]]],
    model: Optional[str],
) -> Iterator[str]:
    if not model_input:
        return
    if isinstance(model_input, str):
        text = model_input
        yield text
        return
    if not isinstance(model_input, Sequence):
        return
    if any(not isinstance(item, str) for item in model_input):
        # FIXME: We can't decode tokens (List[int]) reliably because the model name is not reliable,
        # e.g. for text-embedding-ada-002 (cl100k_base), OpenAI returns "text-embedding-ada-002-v2",
        # and Azure returns "ada", which refers to a different model (r50k_base). We could use the
        # request model name instead, but that doesn't work for Azure because Azure uses the
        # deployment name (which differs from the model name).
        return
    for text in cast(List[str], model_input):
        yield text
