from openinference.instrumentation.helpers import str_span_id, str_trace_id
from opentelemetry.trace import INVALID_SPAN


def test_str_span_id() -> None:
    assert str_span_id(INVALID_SPAN) == "0000000000000000"


def test_str_trace_id() -> None:
    assert str_trace_id(INVALID_SPAN) == "00000000000000000000000000000000"
