from typing import Callable

from opentelemetry.trace import Span

from openinference.semconv.trace import SpanAttributes

from .config import TraceConfig
from .logging import logger


def finalize_deferred_input_value(span: Span, serialize: Callable[[], str]) -> None:
    config = getattr(span, "_self_config", None)
    if not isinstance(config, TraceConfig):
        return
    if not span.is_recording() or config.hide_inputs:
        return
    try:
        value = serialize()
    except Exception:
        logger.exception("Failed to resolve deferred input value")
        return
    span.set_attribute(SpanAttributes.INPUT_VALUE, value)
