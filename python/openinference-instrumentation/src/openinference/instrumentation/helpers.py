from opentelemetry.trace import Span


def get_span_id(span: Span) -> str:
    return span.get_span_context().span_id.to_bytes(8, "big").hex()


def get_trace_id(span: Span) -> str:
    return span.get_span_context().trace_id.to_bytes(16, "big").hex()
