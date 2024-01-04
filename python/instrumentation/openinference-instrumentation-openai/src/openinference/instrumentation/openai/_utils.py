import json
import logging
import warnings
from enum import Enum
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

from openinference.instrumentation.openai._with_span import _WithSpan
from openinference.semconv.trace import SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes, AttributeValue

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_OPENAI_VERSION = tuple(map(int, version("openai").split(".")[:3]))


class _MimeType(Enum):
    text_plain = "text/plain"
    application_json = "application/json"


class _ValueAndType(NamedTuple):
    value: str
    type: _MimeType


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
            return _ValueAndType(value, _MimeType.application_json)
    if not isinstance(obj, str) and isinstance(obj, (Sequence, Mapping)):
        try:
            value = json.dumps(obj)
        except Exception:
            logger.exception("Failed to dump json")
        else:
            return _ValueAndType(value, _MimeType.application_json)
    return _ValueAndType(str(obj), _MimeType.text_plain)


def _as_input_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.INPUT_VALUE, value_and_type.value
    yield SpanAttributes.INPUT_MIME_TYPE, value_and_type.type.value


def _as_output_attributes(
    value_and_type: Optional[_ValueAndType],
) -> Iterator[Tuple[str, AttributeValue]]:
    if not value_and_type:
        return
    yield SpanAttributes.OUTPUT_VALUE, value_and_type.value
    yield SpanAttributes.OUTPUT_MIME_TYPE, value_and_type.type.value


class _HasAttributes(Protocol):
    def get_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        ...

    def get_extra_attributes(self) -> Iterator[Tuple[str, AttributeValue]]:
        ...


def _finish_tracing(
    with_span: _WithSpan,
    has_attributes: _HasAttributes,
    status_code: Optional[trace_api.StatusCode] = None,
) -> None:
    try:
        attributes: Attributes = dict(has_attributes.get_attributes())
    except Exception:
        logger.exception("Failed to get output value")
        attributes = None
    try:
        extra_attributes: Attributes = dict(has_attributes.get_extra_attributes())
    except Exception:
        logger.exception("Failed to get extra attributes")
        extra_attributes = None
    try:
        with_span.finish_tracing(
            status_code=status_code,
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
