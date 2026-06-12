from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.strands_agents.processor import StrandsAgentsToOpenInferenceProcessor

SENTINEL = "REPRO_BUG_SENTINEL"

def test_repro():
    # Bug: StrandsAgentsToOpenInferenceProcessor.on_end() runs on ALL spans with no framework guard,
    # destroying attributes of spans that do not belong to the framework.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(StrandsAgentsToOpenInferenceProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("non-framework-span") as span:
        span.set_attribute("custom.key", "preserved_value")
        span.set_attribute("http.method", "GET")

    spans = exporter.get_finished_spans()
    assert spans, f'REPRO: {SENTINEL} — no spans recorded'
    attrs = dict(spans[0].attributes or {})
    assert attrs.get('custom.key') == 'preserved_value', (
        f'REPRO: {SENTINEL} — {processorClass}.on_end() destroyed non-framework span attributes. '
        f'Expected custom.key=preserved_value but got: {attrs}'
    )