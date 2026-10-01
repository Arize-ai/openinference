"""
Call-site helper that writes ``input.value`` after ``OITracer.start_span``.

OpenTelemetry sampling and ``TraceConfig.mask`` run inside ``start_span`` and
accept only OTel ``AttributeValue`` types: strings, numbers, booleans, and
homogeneous sequences. A serializer that walks a request body or uploads
blobs is not an ``AttributeValue``, so it cannot be passed at start.

Instrumentors plant a masked placeholder such as ``REDACTED_VALUE`` on
``SpanAttributes.INPUT_VALUE`` at start so the sampler still sees a
side-effect-free input. After start, ``finalize_deferred_input_value``
takes a zero-argument serializer. The helper invokes that serializer only
when the span is recording and ``TraceConfig.hide_inputs`` is false. It
then writes the result as ``input.value``. Dropped, suppressed, ended, and
non-OpenInference spans leave the placeholder and never call the serializer.

If ``serialize`` raises, the helper logs the exception and leaves the
placeholder in place.
"""

from typing import Callable

from opentelemetry.trace import Span

from openinference.semconv.trace import SpanAttributes

from .config import TraceConfig
from .logging import logger


def finalize_deferred_input_value(span: Span, serialize: Callable[[], str]) -> None:
    """
    Resolve a deferred ``input.value`` and set it on ``span`` when it is safe
    to run the serializer.

    This is a no-op when ``span`` is not an OpenInference span, is not
    recording, or has ``hide_inputs`` set. If ``serialize`` raises, the helper
    logs the exception and leaves existing attributes unchanged.

    Args:
        span: Span returned by ``OITracer.start_span`` or
            ``start_as_current_span``.
        serialize: Zero-argument callable that returns the final
            ``input.value`` string. Called at most once, and only on the
            recording path.
    """
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
