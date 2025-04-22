from typing import Iterator

import openai
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import SpanAttributes


class TestTokenCounts:
    # @pytest.mark.
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    def test_openai(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        client = openai.OpenAI(api_key="sk-")
        resp = client.chat.completions.create(
            extra_headers={"Accept-Encoding": "gzip"},
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        usage = resp.usage
        assert usage is not None

        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})

        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == usage.completion_tokens
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == usage.prompt_tokens
        assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == usage.total_tokens

        # Check for detailed token stats if available in the response
        if hasattr(usage, "prompt_tokens_details"):
            assert usage.prompt_tokens_details is not None
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ)
                == usage.prompt_tokens_details.cached_tokens
            )
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO)
                == usage.prompt_tokens_details.audio_tokens
            )

        if hasattr(usage, "completion_tokens_details"):
            assert usage.completion_tokens_details is not None
            assert (
                attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO)
                == usage.completion_tokens_details.audio_tokens
            )
            assert (
                attr.pop(LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING)
                == usage.completion_tokens_details.reasoning_tokens
            )


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    OpenAIInstrumentor().uninstrument()


LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO
LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING = (
    SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
)
