from typing import Any, Dict

from google.cloud.aiplatform_v1beta1.types import GenerateContentResponse

from openinference.instrumentation.vertexai._wrapper import _parse_usage_metadata


def test_usage_metadata_includes_cached_and_thoughts() -> None:
    usage = GenerateContentResponse.UsageMetadata(
        prompt_token_count=100,
        candidates_token_count=20,
        total_token_count=150,
        cached_content_token_count=60,
        thoughts_token_count=30,
    )
    attrs: Dict[str, Any] = dict(_parse_usage_metadata(usage))
    assert attrs == {
        "llm.token_count.prompt": 100,
        "llm.token_count.prompt_details.cache_read": 60,
        "llm.token_count.completion": 50,
        "llm.token_count.completion_details.reasoning": 30,
        "llm.token_count.total": 150,
    }


def test_usage_metadata_without_cached_or_thoughts() -> None:
    usage = GenerateContentResponse.UsageMetadata(
        prompt_token_count=10, candidates_token_count=5, total_token_count=15
    )
    assert dict(_parse_usage_metadata(usage)) == {
        "llm.token_count.prompt": 10,
        "llm.token_count.completion": 5,
        "llm.token_count.total": 15,
    }
