from asyncio import create_task, gather, sleep
from random import random

from llama_index.core.instrumentation import get_dispatcher
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor, get_current_span
from openinference.semconv.trace import SpanAttributes

dispatcher = get_dispatcher(__name__)


@dispatcher.span  # type: ignore[misc,unused-ignore]
async def foo(k: int = 1) -> str:
    child = create_task(foo(k - 1)) if k > 1 else None
    await sleep(random() / 100)
    span = get_current_span()
    if child:
        await child
    return str(span.get_span_context().span_id) if span else ""


async def test_get_current_span(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    assert await foo() == ""
    n, k = 10, 5
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    await gather(*(foo(k) for _ in range(n)))
    LlamaIndexInstrumentor().uninstrument()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == n * k
    seen = set()
    for span in spans:
        assert span.attributes and span.context
        assert (expected := str(span.context.span_id)) not in seen
        seen.add(expected)
        assert span.attributes.get(OUTPUT_VALUE) == expected


OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
