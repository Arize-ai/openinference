import pytest
from llama_index.core.instrumentation import get_dispatcher  # type: ignore[attr-defined]
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

dispatcher = get_dispatcher(__name__)


@dispatcher.span  # type: ignore[misc,unused-ignore]
def foo() -> None: ...


@pytest.mark.parametrize("separate_trace_from_runtime_context", [True, False])
def test_separate_trace_from_runtime_context(
    separate_trace_from_runtime_context: bool,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    LlamaIndexInstrumentor().instrument(
        tracer_provider=tracer_provider,
        separate_trace_from_runtime_context=separate_trace_from_runtime_context,
    )
    with tracer_provider.get_tracer(__name__).start_as_current_span("parent"):
        foo()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    assert spans[0].name == "foo"
    if separate_trace_from_runtime_context:
        assert spans[0].parent is None
    else:
        assert spans[0].parent
        assert spans[1].context
        assert spans[0].parent.span_id == spans[1].context.span_id
