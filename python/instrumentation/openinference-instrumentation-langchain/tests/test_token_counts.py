import pytest
from langchain_core.messages.ai import UsageMetadata

from openinference.instrumentation.langchain._tracer import (
    _token_counts_from_lc_usage_metadata,
    _token_counts_from_raw_anthropic_usage_with_cache,
    _token_counts_from_raw_anthropic_usage_with_cache_creation,
    _token_counts_from_raw_anthropic_usage_with_cache_read,
)
from openinference.semconv.trace import SpanAttributes


@pytest.mark.parametrize(
    "usage_metadata,expected",
    [
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="basic_token_counts",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 35,
                "input_token_details": {"audio": 5},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 35,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: 5,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 15,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="input_audio_details",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 38,
                "input_token_details": {"cache_creation": 8},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 38,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 8,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 18,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="input_cache_creation_details",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 33,
                "input_token_details": {"cache_read": 3},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 33,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 3,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 13,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="input_cache_read_details",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 37,
                "output_token_details": {"audio": 7},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 37,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: 7,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 27,
            },
            id="output_audio_details",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 35,
                "output_token_details": {"reasoning": 5},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 35,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: 5,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 25,
            },
            id="output_reasoning_details",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 50,
                "input_token_details": {
                    "audio": 5,
                    "cache_creation": 3,
                    "cache_read": 2,
                },
                "output_token_details": {
                    "audio": 6,
                    "reasoning": 4,
                },
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 50,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: 5,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 3,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 2,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: 6,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: 4,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 20,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 30,
            },
            id="all_details",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_token_details": {"audio": 5},
                "output_token_details": {"reasoning": 3},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: 5,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: 3,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="total_equals_sum_no_detail_adjustment",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_token_details": {},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="empty_input_token_details",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_token_details": {},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="empty_output_token_details",
        ),
        pytest.param(
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 0,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 0,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 0,
            },
            id="zero_values",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_token_details": {
                    "audio": None,
                    "cache_creation": 5,
                },
                "output_token_details": {
                    "audio": 3,
                    "reasoning": None,
                },
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 5,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO: 3,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="none_values_not_yielded",
        ),
    ],
)
def test_token_counts_from_lc_usage_metadata(
    usage_metadata: UsageMetadata, expected: dict[str, int]
) -> None:
    """Test _token_counts_from_lc_usage_metadata with various inputs."""
    result = dict(_token_counts_from_lc_usage_metadata(usage_metadata))
    assert result == expected


@pytest.mark.parametrize(
    "usage,expected",
    [
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 3,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 5,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 3,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 18,  # 10 + 5 + 3
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            id="with_both_cache_types",
        ),
        pytest.param(
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 0,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 0,
            },
            id="zeros_no_cache_details",
        ),
    ],
)
def test_token_counts_from_raw_anthropic_usage_with_cache(
    usage: dict[str, int], expected: dict[str, int]
) -> None:
    """Test Anthropic usage with both cache creation and read."""
    result = dict(_token_counts_from_raw_anthropic_usage_with_cache(usage))  # type: ignore
    assert result == expected


@pytest.mark.parametrize(
    "usage,expected",
    [
        pytest.param(
            {
                "input_tokens": 15,
                "output_tokens": 25,
                "cache_creation_input_tokens": 8,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 8,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 23,  # 15 + 8
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 25,
            },
            id="with_cache_creation",
        ),
    ],
)
def test_token_counts_from_raw_anthropic_usage_with_cache_creation(
    usage: dict[str, int], expected: dict[str, int]
) -> None:
    """Test Anthropic usage with cache creation only."""
    result = dict(_token_counts_from_raw_anthropic_usage_with_cache_creation(usage))  # type: ignore
    assert result == expected


@pytest.mark.parametrize(
    "usage,expected",
    [
        pytest.param(
            {
                "input_tokens": 12,
                "output_tokens": 18,
                "cache_read_input_tokens": 6,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 6,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 18,  # 12 + 6
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 18,
            },
            id="with_cache_read",
        ),
    ],
)
def test_token_counts_from_raw_anthropic_usage_with_cache_read(
    usage: dict[str, int], expected: dict[str, int]
) -> None:
    """Test Anthropic usage with cache read only."""
    result = dict(_token_counts_from_raw_anthropic_usage_with_cache_read(usage))  # type: ignore
    assert result == expected
