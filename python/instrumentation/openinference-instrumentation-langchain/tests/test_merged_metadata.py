import json

from langchain_core.runnables import RunnableConfig, RunnableLambda
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation import using_metadata
from openinference.semconv.trace import SpanAttributes


async def test_merged_metadata(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with using_metadata({"b": "2", "c": "3"}):
        RunnableLambda(lambda _: None).invoke(..., RunnableConfig(metadata={"a": 1, "b": 2}))
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "RunnableLambda"
    attributes = dict(spans[0].attributes or {})
    assert isinstance(metadata_str := attributes.get(SpanAttributes.METADATA), str)
    assert json.loads(metadata_str) == {"a": 1, "b": 2, "c": "3"}
