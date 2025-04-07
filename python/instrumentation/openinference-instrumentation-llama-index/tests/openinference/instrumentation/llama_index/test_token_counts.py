from typing import Iterator

import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.groq import Groq  # type: ignore[import-untyped]
from llama_index.llms.openai import OpenAI
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.semconv.trace import SpanAttributes


class TestTokenCounts:
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    async def test_groq(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        result = await Groq(
            model="llama3-8b-8192",
            api_key="gsk_",
        ).astream_chat([ChatMessage(content="Hello!")])
        async for _ in result:
            pass
        span = in_memory_span_exporter.get_finished_spans()[0]
        assert span.attributes
        assert span.attributes.get(LLM_TOKEN_COUNT_TOTAL)
        assert span.attributes.get(LLM_TOKEN_COUNT_COMPLETION)
        assert span.attributes.get(LLM_TOKEN_COUNT_TOTAL)

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    def test_openai(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        llm = OpenAI(model="gpt-4o-mini", api_key="sk-")
        resp = llm.chat([ChatMessage(content="Hello!")])
        usage = resp.raw.usage

        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})

        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == resp.raw.usage.completion_tokens
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == resp.raw.usage.prompt_tokens
        assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == resp.raw.usage.total_tokens
        assert (
            attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ)
            == usage.prompt_tokens_details.cached_tokens
        )
        assert (
            attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO)
            == usage.prompt_tokens_details.audio_tokens
        )
        assert (
            attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO)
            == usage.completion_tokens_details.audio_tokens
        )
        assert (
            attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING)
            == usage.completion_tokens_details.reasoning_tokens
        )

    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    def test_anthropic(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        llm = Anthropic(model="claude-3-5-haiku-20241022", api_key="sk-")
        resp = llm.chat([ChatMessage(content="Hello!")])
        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})

        usage = resp.raw.get("usage")
        assert (
            attr.pop(LLM_TOKEN_COUNT_PROMPT)
            == usage.input_tokens
            + usage.cache_creation_input_tokens
            + usage.cache_read_input_tokens
        )
        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == usage.output_tokens
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == usage.cache_read_input_tokens
        assert (
            attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE)
            == usage.cache_creation_input_tokens
        )


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
LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO
LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING = (
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
)
