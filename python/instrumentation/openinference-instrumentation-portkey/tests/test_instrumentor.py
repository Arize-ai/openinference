from importlib import import_module

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from openinference.semconv.trace import MessageAttributes, SpanAttributes

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
        messages=[{"role": "user", "content": "What's the weather like?"}], model="gpt-4o-mini"
    )
    print(resp)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})

    expected_attributes = {
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "user",
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "What's the weather like?",
        SpanAttributes.OUTPUT_MIME_TYPE: 'application/json',
        SpanAttributes.INPUT_MIME_TYPE: 'application/json',
        SpanAttributes.LLM_MODEL_NAME: 'gpt-4o-mini-2024-07-18',
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 63,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 12,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 51,
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}": "assistant",
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}": "I don't have real-time data access to provide current weather updates. However, you can check a weather website or app for the latest information in your area. If you tell me your location, I can suggest typical weather patterns for this time of year!",
        SpanAttributes.OPENINFERENCE_SPAN_KIND: 'LLM',
    }

    for key, expected_value in expected_attributes.items():
        assert attributes.get(key) == expected_value
