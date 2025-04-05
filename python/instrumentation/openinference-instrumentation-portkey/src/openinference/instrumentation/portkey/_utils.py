import logging
from typing import Any, Dict, Optional, Tuple

from opentelemetry import trace as trace_api
from opentelemetry.trace import Span

from openinference.semconv.trace import SpanAttributes

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _finish_tracing(
    span: Optional[Span],
    response: Any,
    error: Optional[Exception] = None,
) -> None:
    """Finish tracing a span with response and error information."""
    if span is None:
        return

    if error is not None:
        span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))
        span.record_exception(error)
        span.set_attribute(SpanAttributes.ERROR_TYPE, type(error).__name__)
        span.set_attribute(SpanAttributes.ERROR_MESSAGE, str(error))
    else:
        span.set_status(trace_api.Status(trace_api.StatusCode.OK))


def _extract_model_name(model: Any) -> str:
    """Extract the model name from the model parameter."""
    if isinstance(model, str):
        return model
    if hasattr(model, "model"):
        return str(model.model)
    return str(model)


def _extract_model_version(model: Any) -> Optional[str]:
    """Extract the model version from the model parameter."""
    if hasattr(model, "version"):
        return str(model.version)
    return None


def _extract_model_provider(model: Any) -> Optional[str]:
    """Extract the model provider from the model parameter."""
    if hasattr(model, "provider"):
        return str(model.provider)
    return None


def _extract_model_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model parameters from the parameters dictionary."""
    # Filter out None values and convert to strings
    return {k: str(v) for k, v in params.items() if v is not None} 