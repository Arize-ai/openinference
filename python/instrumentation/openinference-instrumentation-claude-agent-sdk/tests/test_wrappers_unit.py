"""Unit tests for helpers in `_wrappers.py` that don't need the full instrumentor.

Covers `_extract_model_name_from_usage` (regression coverage for issue #3136,
where the helper returned the first dict key instead of the model that actually
did the bulk of the work in a multi-model run) and
`_extract_usage_and_cost_attributes` (regression coverage for the cache-token
undercount: Anthropic's `input_tokens` excludes cache tokens by design, so
LLM_TOKEN_COUNT_PROMPT/TOTAL must fold cache_read_input_tokens and
cache_creation_input_tokens back in).
"""

from __future__ import annotations

from typing import Any

import pytest

from openinference.instrumentation.claude_agent_sdk._wrappers import (
    _extract_model_name_from_usage,
    _extract_usage_and_cost_attributes,
)
from openinference.semconv.trace import SpanAttributes

# ---------------------------------------------------------------------------
# Mapping-shaped `modelUsage` — the case #3136 was about
# ---------------------------------------------------------------------------


def test_single_model_dict_returns_that_model() -> None:
    usage = {
        "claude-sonnet-4-6": {
            "outputTokens": 4,
            "inputTokens": 3,
            "costUSD": 0.008627,
        }
    }
    assert _extract_model_name_from_usage(usage) == "claude-sonnet-4-6"


def test_multi_model_dict_picks_max_output_tokens() -> None:
    # The fast/router model emits a tiny number of tokens; the main model does
    # the bulk of the generation. The span attribute should reflect the latter.
    usage = {
        "claude-haiku-4-5": {"outputTokens": 5, "inputTokens": 200},
        "claude-sonnet-4-6": {"outputTokens": 350, "inputTokens": 8},
    }
    assert _extract_model_name_from_usage(usage) == "claude-sonnet-4-6"


def test_multi_model_dict_picks_max_output_tokens_irrespective_of_dict_order() -> None:
    # Same shape, opposite insertion order — must still pick the heavy-output model.
    usage = {
        "claude-sonnet-4-6": {"outputTokens": 350, "inputTokens": 8},
        "claude-haiku-4-5": {"outputTokens": 5, "inputTokens": 200},
    }
    assert _extract_model_name_from_usage(usage) == "claude-sonnet-4-6"


def test_snake_case_output_tokens_also_accepted() -> None:
    # Some SDK shapes use snake_case; both should be treated as the same field.
    usage = {
        "claude-haiku-4-5": {"output_tokens": 5},
        "claude-sonnet-4-6": {"output_tokens": 400},
    }
    assert _extract_model_name_from_usage(usage) == "claude-sonnet-4-6"


def test_missing_output_tokens_falls_back_to_zero_weight() -> None:
    # If neither key has an outputTokens field, max() falls back to 0 weight for
    # every entry and the first key (by max-of-equals) is returned — but the
    # function must not crash.
    usage = {
        "model-a": {"inputTokens": 10},
        "model-b": {"inputTokens": 20},
    }
    # Either model is acceptable behavior; just assert no crash + a real name.
    result = _extract_model_name_from_usage(usage)
    assert result in ("model-a", "model-b")


def test_non_mapping_entry_value_does_not_crash() -> None:
    # Defensive: some SDK versions might pass a string or None alongside a dict.
    usage = {
        "model-a": "unexpected-string-value",
        "model-b": {"outputTokens": 99},
    }
    assert _extract_model_name_from_usage(usage) == "model-b"


def test_non_int_output_tokens_does_not_crash() -> None:
    usage = {
        "model-a": {"outputTokens": "not-a-number"},
        "model-b": {"outputTokens": 50},
    }
    assert _extract_model_name_from_usage(usage) == "model-b"


def test_empty_dict_returns_none() -> None:
    assert _extract_model_name_from_usage({}) is None


# ---------------------------------------------------------------------------
# Non-mapping shapes — must still work as before (regression for the
# list / object fallback branches the original function had)
# ---------------------------------------------------------------------------


def test_list_of_entries_returns_first_named() -> None:
    usage = [
        {"model": "claude-sonnet-4-6", "outputTokens": 10},
        {"model": "claude-haiku-4-5", "outputTokens": 200},
    ]
    # List branch keeps its original "first named entry" semantics — this PR
    # only changes the dict branch. Calling out the deliberate divergence.
    assert _extract_model_name_from_usage(usage) == "claude-sonnet-4-6"


def test_object_with_model_attribute() -> None:
    class FakeUsage:
        model = "claude-sonnet-4-6"

    assert _extract_model_name_from_usage(FakeUsage()) == "claude-sonnet-4-6"


def test_none_returns_none() -> None:
    assert _extract_model_name_from_usage(None) is None


@pytest.mark.parametrize("value", ["", 0, [], {}])
def test_falsy_inputs_return_none(value: object) -> None:
    assert _extract_model_name_from_usage(value) is None


# ---------------------------------------------------------------------------
# `_extract_usage_and_cost_attributes` — cache tokens fold into prompt/total.
# Anthropic's input_tokens excludes cache tokens by design (same Messages-API
# `usage` shape the sibling openinference-instrumentation-anthropic package
# instruments), so prompt/total must add cache_read/cache_creation back in.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "usage,expected_prompt,expected_completion,expected_total",
    [
        pytest.param(
            # Real usage payload recorded in this package's own cassette
            # (tests/cassettes/test_instrumentor/test_query_real_agent_span.yaml).
            {
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_read_input_tokens": 17024,
                "cache_creation_input_tokens": 0,
            },
            17027,
            4,
            17031,
            id="cache_read_folded_into_prompt_and_total",
        ),
        pytest.param(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 500,
            },
            510,
            5,
            515,
            id="cache_creation_folded_into_prompt_and_total",
        ),
        pytest.param(
            {"input_tokens": 12, "output_tokens": 8},
            12,
            8,
            20,
            id="no_cache_tokens_leaves_prompt_and_total_unchanged",
        ),
        pytest.param(
            # input_tokens absent but cache tokens known: prompt must still be
            # emitted so it stays a superset of the prompt_details breakouts.
            {"output_tokens": 4, "cache_read_input_tokens": 17024},
            17024,
            4,
            17028,
            id="cache_only_payload_still_yields_prompt_and_total",
        ),
        pytest.param(
            # output_tokens absent (error/aborted results): total = prompt,
            # matching the sibling anthropic package's prompt + (output or 0).
            {"input_tokens": 3, "cache_read_input_tokens": 17024},
            17027,
            None,
            17027,
            id="missing_output_tokens_still_yields_total",
        ),
        pytest.param(
            # Present-but-unparseable cache_write_input_tokens must fall back
            # to cache_creation_input_tokens rather than dropping the count.
            {
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_write_input_tokens": {"ephemeral_5m_input_tokens": 500},
                "cache_creation_input_tokens": 500,
            },
            503,
            4,
            507,
            id="unparseable_cache_write_falls_back_to_cache_creation",
        ),
    ],
)
def test_token_counts_fold_cache_tokens(
    usage: dict[str, Any],
    expected_prompt: int,
    expected_completion: int | None,
    expected_total: int,
) -> None:
    attrs = _extract_usage_and_cost_attributes({"usage": usage})
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == expected_prompt
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == expected_completion
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == expected_total


def test_cache_breakouts_are_unaffected_by_the_fold_in() -> None:
    attrs = _extract_usage_and_cost_attributes(
        {
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_read_input_tokens": 17024,
                "cache_creation_input_tokens": 100,
            }
        }
    )
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 17024
    assert attrs[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE] == 100


def test_empty_usage_sets_no_token_attributes() -> None:
    attrs = _extract_usage_and_cost_attributes({"usage": {}})
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in attrs
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in attrs
