from openinference.instrumentation.helpers import get_span_id, get_trace_id
from opentelemetry.trace import INVALID_SPAN


def test_get_span_id() -> None:
    assert get_span_id(INVALID_SPAN) == "0000000000000000"


def test_get_trace_id() -> None:
    assert get_trace_id(INVALID_SPAN) == "00000000000000000000000000000000"
