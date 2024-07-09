from openinference.instrumentation.helpers import get_span_id, get_trace_id
from opentelemetry.trace import NonRecordingSpan, SpanContext


def test_get_span_and_trace_ids() -> None:
    span = NonRecordingSpan(
        SpanContext(
            trace_id=83298525428589810613581548256892812612,
            span_id=8006530202346048876,
            is_remote=False,
        )
    )
    assert get_span_id(span) == "6f1ce8cc7245cd6c"
    assert get_trace_id(span) == "3eaab662c550df264f0fbd19bd8bfd44"
