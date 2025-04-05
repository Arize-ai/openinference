import json
from importlib import import_module
from typing import Tuple, cast

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry import trace as trace_api

@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_chat_completion(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
) -> None:
    portkey = import_module("portkey_ai")
    client = portkey.Portkey(
        provider="openai",
        Authorization="sk-***"
    )
    client.chat.completions.create(
        messages=[{"role": "user", "content": "What's the weather like?"}],
        model="gpt-4o-mini"
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    assert attributes.pop("llm.input_messages.0.message.role") == "user"
    assert attributes.pop("llm.input_messages.0.message.content") == "What's the weather like?"