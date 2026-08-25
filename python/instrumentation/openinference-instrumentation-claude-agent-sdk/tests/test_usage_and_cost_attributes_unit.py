"""Unit tests for `_extract_usage_and_cost_attributes` in `_wrappers.py`.

Regression coverage for the reported cache-token undercount: `input_tokens` in
the Claude Agent SDK's usage payload excludes cache tokens by design (same
Anthropic Messages API `usage` shape the sibling `openinference-instrumentation-
anthropic` package instruments), so LLM_TOKEN_COUNT_PROMPT/TOTAL must fold
cache_read_input_tokens and cache_creation_input_tokens back in.
"""

from __future__ import annotations

from openinference.instrumentation.claude_agent_sdk._wrappers import (
    _extract_usage_and_cost_attributes,
)
from openinference.semconv.trace import SpanAttributes


def test_cache_read_tokens_are_folded_into_prompt_and_total() -> None:
    # Real usage payload recorded in this package's own cassette
    # (tests/cassettes/test_instrumentor/test_query_real_agent_span.yaml).
    msg = {
        "usage": {
            "input_tokens": 3,
            "output_tokens": 4,
            "cache_read_input_tokens": 17024,
            "cache_creation_input_tokens": 0,
        }
    }
    attrs = _extract_usage_and_cost_attributes(msg)
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 17027
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 17031
    # The separate cache-read breakout attribute is unaffected by the fold-in.
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 17024


def test_cache_creation_tokens_are_also_folded_into_prompt() -> None:
    msg = {
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 500,
        }
    }
    attrs = _extract_usage_and_cost_attributes(msg)
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 510
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 515


def test_no_cache_tokens_leaves_prompt_and_total_unchanged() -> None:
    msg = {"usage": {"input_tokens": 12, "output_tokens": 8}}
    attrs = _extract_usage_and_cost_attributes(msg)
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 12
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 20
