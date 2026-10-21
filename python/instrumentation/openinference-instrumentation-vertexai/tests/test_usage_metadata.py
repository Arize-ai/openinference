import pytest
from google.cloud import aiplatform_v1 as v1
from google.cloud import aiplatform_v1beta1 as v1beta1

from openinference.instrumentation.vertexai._wrapper import _parse_usage_metadata
from openinference.semconv.trace import SpanAttributes


@pytest.mark.parametrize("module", [v1, v1beta1], ids=["v1", "v1beta1"])
def test_usage_metadata_without_cache_or_thoughts(module) -> None:  # type: ignore[no-untyped-def]
    usage_metadata = module.GenerateContentResponse.UsageMetadata(
        prompt_token_count=100,
        candidates_token_count=20,
        total_token_count=120,
    )
    assert dict(_parse_usage_metadata(usage_metadata)) == {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 100,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 120,
    }


def test_usage_metadata_records_cache_read_and_thoughts() -> None:
    usage_metadata = v1beta1.GenerateContentResponse.UsageMetadata(
        prompt_token_count=100,
        candidates_token_count=20,
        thoughts_token_count=300,
        cached_content_token_count=60,
        total_token_count=420,
    )
    assert dict(_parse_usage_metadata(usage_metadata)) == {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 100,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 60,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 320,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: 300,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 420,
    }
