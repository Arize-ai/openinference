from opentelemetry.sdk.trace import ReadableSpan
from openinference.semconv.trace import SpanAttributes
def is_openinference_span(span: ReadableSpan) -> bool:
    """Check if a span is an OpenInference span."""
    if span.attributes is None:
        return False
    return SpanAttributes.OPENINFERENCE_SPAN_KIND in span.attributes