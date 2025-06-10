import pytest
from langchain_core.runnables import RunnableLambda
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.langchain import LangChainInstrumentor


@pytest.mark.parametrize("separate_trace_from_runtime_context", [True, False])
def test_separate_trace_from_runtime_context(
    separate_trace_from_runtime_context: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    LangChainInstrumentor().uninstrument()
    LangChainInstrumentor().instrument(
        tracer_provider=tracer_provider,
        separate_trace_from_runtime_context=separate_trace_from_runtime_context,
    )
    with tracer_provider.get_tracer(__name__).start_as_current_span("parent"):
        RunnableLambda(lambda _: None).invoke(...)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].name == "RunnableLambda"
    if separate_trace_from_runtime_context:
        assert spans[0].parent is None
    else:
        assert spans[0].parent
        assert spans[1].context
        assert spans[0].parent.span_id == spans[1].context.span_id
