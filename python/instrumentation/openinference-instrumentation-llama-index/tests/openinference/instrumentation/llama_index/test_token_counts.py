from typing import Iterator

import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.groq import Groq  # type: ignore[import-untyped]
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import SpanAttributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
async def test_groq_astream_chat_token_count(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    result = await Groq(model="llama3-8b-8192").astream_chat([ChatMessage(content="Hello!")])
    async for _ in result:
        pass
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes
    assert span.attributes.get(LLM_TOKEN_COUNT_TOTAL)
    assert span.attributes.get(LLM_TOKEN_COUNT_COMPLETION)
    assert span.attributes.get(LLM_TOKEN_COUNT_TOTAL)


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    LlamaIndexInstrumentor().uninstrument()


LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
