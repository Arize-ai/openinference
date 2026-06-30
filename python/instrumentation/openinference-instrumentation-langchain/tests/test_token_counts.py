from typing import Any

import pytest

from openinference.instrumentation.langchain._tracer import (
    _is_lc_usage_metadata,
    _is_raw_anthropic_usage_with_cache_read_or_write,
    _token_counts_from_lc_usage_metadata,
    _token_counts_from_raw_anthropic_usage_with_cache_read_or_write,
)
from openinference.semconv.trace import SpanAttributes


@pytest.mark.parametrize(
    "usage_metadata,expected,is_valid",
    [
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
            },
            True,
            id="basic",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 35,
                "input_token_details": {"cache_creation": 3, "cache_read": 2},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 15,  # 10 + 3 + 2
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 35,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 3,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 2,
            },
            True,
            id="bedrock_converse",
        ),
        pytest.param(
            {
                "input_tokens": 5,
                "output_tokens": 10,
                "total_tokens": 15,
                "input_token_details": {"cache_creation": 20, "cache_read": 10},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 35,  # 5 + 20 + 10
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 10,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 45,  # adjusted
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 20,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 10,
            },
            True,
            id="bedrock_invokemodel",
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
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO: 5,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING: 3,
            },
            True,
            id="non_cache_details",
        ),
        pytest.param(
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 0,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 0,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 0,
            },
            True,
            id="zeros",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_token_details": {"cache_creation": 0, "cache_read": 0},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
            },
            True,
            id="zero_cache_no_details",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "input_token_details": {},
                "output_token_details": {},
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 30,
            },
            True,
            id="empty_details",
        ),
        pytest.param(
            {"input_tokens": 10, "output_tokens": 20},
            {},
            False,
            id="missing_total",
        ),
        pytest.param(
            {"input_tokens": "10", "output_tokens": 20, "total_tokens": 30},
            {},
            False,
            id="wrong_type",
        ),
        pytest.param(
            {"output_tokens": 20, "total_tokens": 30},
            {},
            False,
            id="missing_field",
        ),
    ],
)
def test_token_counts_from_lc_usage_metadata(
    usage_metadata: dict[str, Any], expected: dict[str, int], is_valid: bool
) -> None:
    """Test _token_counts_from_lc_usage_metadata with various inputs."""
    assert _is_lc_usage_metadata(usage_metadata) == is_valid
    if _is_lc_usage_metadata(usage_metadata):
        result = dict(_token_counts_from_lc_usage_metadata(usage_metadata))
        assert result == expected


@pytest.mark.parametrize(
    "usage,expected,is_valid",
    [
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 5,
                "cache_read_input_tokens": 3,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 18,  # 10 + 5 + 3
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 5,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 3,
            },
            True,
            id="both",
        ),
        pytest.param(
            {"input_tokens": 15, "output_tokens": 25, "cache_creation_input_tokens": 8},
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 23,  # 15 + 8
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 25,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 8,
            },
            True,
            id="write",
        ),
        pytest.param(
            {"input_tokens": 12, "output_tokens": 18, "cache_read_input_tokens": 6},
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 18,  # 12 + 6
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 18,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 6,
            },
            True,
            id="read",
        ),
        pytest.param(
            {"input_tokens": 10, "output_tokens": 20, "cache_creation_input_tokens": 0},
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            True,
            id="zero_cache_write",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 20,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
            {
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 10,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 20,
            },
            True,
            id="zero_both_cache",
        ),
        pytest.param(
            {"input_tokens": 10, "output_tokens": 20},
            {},
            False,
            id="no_cache",
        ),
        pytest.param(
            {"input_tokens": "10", "output_tokens": 20, "cache_read_input_tokens": 5},
            {},
            False,
            id="wrong_type",
        ),
        pytest.param(
            {"output_tokens": 20, "cache_read_input_tokens": 5},
            {},
            False,
            id="missing_field",
        ),
    ],
)
def test_token_counts_from_raw_anthropic_usage(
    usage: dict[str, Any], expected: dict[str, int], is_valid: bool
) -> None:
    """Test Anthropic usage with cache."""
    assert _is_raw_anthropic_usage_with_cache_read_or_write(usage) == is_valid
    if _is_raw_anthropic_usage_with_cache_read_or_write(usage):
        result = dict(_token_counts_from_raw_anthropic_usage_with_cache_read_or_write(usage))
        assert result == expected
