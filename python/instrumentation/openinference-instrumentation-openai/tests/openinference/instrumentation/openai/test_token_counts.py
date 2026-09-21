import os
from typing import Optional

import openai
import pytest
from openai.types.completion_usage import CompletionUsage, PromptTokensDetails
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)
from openinference.semconv.trace import OpenInferenceLLMProviderValues, SpanAttributes


class TestTokenCounts:
    # @pytest.mark.
    @pytest.mark.vcr
    def test_openai(
        self,
        in_memory_span_exporter: InMemorySpanExporter,
    ) -> None:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "sk-"))
        resp = client.chat.completions.create(
            extra_headers={"Accept-Encoding": "gzip"},
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello!"}],
        )
        usage = resp.usage
        assert usage is not None

        # HTTPX is instrumented too, so pick the OpenAI LLM span rather than the first span.
        spans = [
            span
            for span in in_memory_span_exporter.get_finished_spans()
            if (span.attributes or {}).get(SpanAttributes.LLM_PROVIDER)
            == OpenInferenceLLMProviderValues.OPENAI.value
        ]
        assert len(spans) == 1
        attr = dict(spans[0].attributes or {})

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


class TestCompletionUsageCacheTokens:
    """Unit tests for cache token extraction from Chat Completions usage.

    ``cache_write_tokens`` is only present on newer OpenAI SDK versions, so the
    usage objects are constructed without validation to keep these tests working
    across SDK versions.
    """

    @pytest.mark.parametrize(
        "prompt_tokens_details,expected",
        [
            pytest.param(
                {"cached_tokens": 0, "cache_write_tokens": 4889},
                {
                    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 0,
                    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 4889,
                },
                id="cache_write_on_first_call",
            ),
            pytest.param(
                {"cached_tokens": 4876, "cache_write_tokens": 13},
                {
                    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 4876,
                    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 13,
                },
                id="cache_read_on_second_call",
            ),
            pytest.param(
                {"cached_tokens": 0, "cache_write_tokens": 0},
                {
                    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 0,
                    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 0,
                },
                id="zero_values_are_still_recorded",
            ),
            pytest.param(
                {"cached_tokens": 7},
                {
                    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 7,
                    LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: None,
                },
                id="older_sdk_without_cache_write_tokens",
            ),
        ],
    )
    def test_get_attributes_from_completion_usage(
        self,
        prompt_tokens_details: dict[str, int],
        expected: dict[str, Optional[int]],
    ) -> None:
        usage = CompletionUsage.model_construct(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            # The leading None is `_fields_set`; without it mypy assumes the splat could fill it.
            prompt_tokens_details=PromptTokensDetails.model_construct(
                None, **prompt_tokens_details
            ),
        )
        extractor = _ResponseAttributesExtractor(openai)
        attributes = dict(extractor._get_attributes_from_completion_usage(usage))

        for key, value in expected.items():
            assert attributes.get(key) == value
