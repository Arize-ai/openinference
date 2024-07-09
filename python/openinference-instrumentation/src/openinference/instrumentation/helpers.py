from opentelemetry.trace import Span


def str_span_id(span: Span) -> str:
    return span.get_span_context().span_id.to_bytes(8, "big").hex()


def str_trace_id(span: Span) -> str:
    return span.get_span_context().span_id.to_bytes(16, "big").hex()
