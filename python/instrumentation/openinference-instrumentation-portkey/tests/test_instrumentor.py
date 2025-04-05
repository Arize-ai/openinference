from importlib import import_module

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry import trace as trace_api

@pytest.mark.vcr(
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_chat_completion(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
    setup_portkey_instrumentation: None,
) -> None:
    portkey = import_module("portkey_ai")
    client = portkey.Portkey(
        api_key="REDACTED",
        virtual_key="REDACTED",
    )
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather like?"}],
        model="gpt-4o-mini"
    )
    print(resp)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})