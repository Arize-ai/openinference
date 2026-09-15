"""Regression test: a mixed-type list payload must not drop the event's attributes."""

import time
from typing import Any, Dict

from llama_index.core.callbacks.schema import CBEventType
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index._callback import _EventData, _finish_tracing


def test_finish_tracing_with_mixed_type_list_payload_keeps_attributes() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(__name__)

    span = tracer.start_span("llama.span")
    key = "llm.input_messages.0.message.contents"
    event = _EventData(
        span=span,
        parent_id=None,
        context=None,
        payloads=[],
        exceptions=[],
        event_type=CBEventType.LLM,
        attributes={
            key: [{"message_content": {"type": "text", "text": "hi"}}, "follow-up"],
            "llm.model_name": "gpt-4o",
        },
        start_time=int(time.time() * 1e9),
        end_time=int(time.time() * 1e9),
    )

    _finish_tracing(event)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})
    assert attributes.get("llm.model_name") == "gpt-4o"
    assert attributes.get(f"{key}.0.message_content.text") == "hi"
    assert attributes.get(f"{key}.1") == "follow-up"
