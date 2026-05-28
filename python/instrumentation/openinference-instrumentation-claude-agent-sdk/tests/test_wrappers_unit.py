"""Unit tests for helpers in `_wrappers.py` that don't need the full instrumentor.

Currently focused on `_extract_model_name_from_usage` (regression coverage for
issue #3136, where the helper returned the first dict key instead of the model
that actually did the bulk of the work in a multi-model run).
"""

from __future__ import annotations

import pytest

from openinference.instrumentation.claude_agent_sdk._wrappers import (
    _extract_model_name_from_usage,
)

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
