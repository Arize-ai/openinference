import json
from typing import Iterator

import litellm
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import SpanAttributes


@pytest.fixture(autouse=True)
def instrument(
    tracer_provider: TracerProvider,
) -> Iterator[None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield


class TestTokenCounts:
    @pytest.mark.vcr(
        decode_compressed_response=True,
        before_record_request=lambda _: _.headers.clear() or _,
        before_record_response=lambda _: {**_, "headers": {}},
    )
    def test_openai(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        messages = [{"role": "user", "content": "Hello!"}]
        resp = litellm.completion(
            model="openai/gpt-4o-mini",
            messages=messages,
            api_key="sk-",
            temperature=0.7,
        )
        usage = resp.usage

        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})
        # make sure we are not leaking any sensitive information
        params_str = attr.get(LLM_INVOCATION_PARAMETERS)
        if params_str is not None:
            params = json.loads(str(params_str))
            assert isinstance(params, dict)
            assert "api_key" not in params

        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == usage.completion_tokens
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == usage.prompt_tokens
        assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == usage.total_tokens

        # Check for detailed token stats if available in the response
        if hasattr(usage, "prompt_tokens_details"):
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ)
                == usage.prompt_tokens_details.cached_tokens
            )
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO)
                == usage.prompt_tokens_details.audio_tokens
            )

        if hasattr(usage, "completion_tokens_details"):
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
        messages = [{"role": "user", "content": "Hello!"}]
        resp = litellm.completion(
            model="anthropic/claude-3-5-haiku-20241022",
            messages=messages,
            api_key="sk-",
        )
        span = in_memory_span_exporter.get_finished_spans()[0]
        attr = dict(span.attributes or {})

        # make sure we are not leaking any sensitive information
        params_str = attr.get(LLM_INVOCATION_PARAMETERS)
        if params_str is not None:
            params = json.loads(str(params_str))
            assert isinstance(params, dict)
            assert "api_key" not in params

        usage = resp.usage
        # Check the token counts litellm always returns
        assert attr.pop(LLM_TOKEN_COUNT_PROMPT) == usage.prompt_tokens
        assert attr.pop(LLM_TOKEN_COUNT_COMPLETION) == usage.completion_tokens
        assert attr.pop(LLM_TOKEN_COUNT_TOTAL) == usage.total_tokens

        # Check additional token counts if present
        if hasattr(usage, "cache_creation_input_tokens"):
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE)
                == usage.cache_creation_input_tokens
            )

        if hasattr(usage, "cache_read_input_tokens"):
            assert (
                attr.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == usage.cache_read_input_tokens
            )


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
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
