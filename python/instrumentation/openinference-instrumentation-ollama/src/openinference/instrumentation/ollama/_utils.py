import logging
from typing import Any, Dict, Iterable, Optional, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import safe_json_dumps
from openinference.instrumentation.ollama._with_span import _WithSpan

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _as_arguments_json(arguments: Any) -> str:
    # Ollama returns tool-call arguments as a mapping, unlike the OpenAI-style
    # JSON string. Serialize mappings so the attribute is always a JSON string.
    if isinstance(arguments, str):
        return arguments
    return safe_json_dumps(arguments)


def _finish_tracing(
    with_span: _WithSpan,
    attributes: Iterable[Tuple[str, AttributeValue]],
    extra_attributes: Iterable[Tuple[str, AttributeValue]],
    status: Optional[trace_api.Status] = None,
) -> None:
    attributes_dict: Optional[Dict[str, AttributeValue]] = None
    extra_attributes_dict: Optional[Dict[str, AttributeValue]] = None
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
