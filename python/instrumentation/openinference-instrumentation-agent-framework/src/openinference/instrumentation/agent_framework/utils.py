from typing import Callable, Optional

from opentelemetry.sdk.trace import ReadableSpan

from openinference.semconv.trace import SpanAttributes

# Define types for span filtering
SpanFilter = Callable[[ReadableSpan], bool]


def is_openinference_span(span: ReadableSpan) -> bool:
    """Check if a span is an OpenInference span."""
    if span.attributes is None:
        return False
    return SpanAttributes.OPENINFERENCE_SPAN_KIND in span.attributes


def should_export_span(span: ReadableSpan, span_filter: Optional[SpanFilter] = None) -> bool:
    """Determine if a span should be exported based on a filter."""
    if span_filter is None:
        return True

    return span_filter(span)
