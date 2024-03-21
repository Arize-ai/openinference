import logging
from typing import (
    Iterator,
    Optional,
    Protocol,
    Tuple,
)

from openinference.instrumentation.mistralai._with_span import _WithSpan
from opentelemetry import trace as trace_api
from opentelemetry.util.types import Attributes, AttributeValue

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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
