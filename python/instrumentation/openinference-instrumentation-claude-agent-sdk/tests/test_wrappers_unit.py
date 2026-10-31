"""Unit tests for helpers in `_wrappers.py` that don't need the full instrumentor.

Covers `_extract_model_name_from_usage` (issue #3136) and
`_extract_usage_and_cost_attributes` (cache-token undercount, #3611).
"""

from __future__ import annotations

from typing import Any

import pytest
from openinference.semconv.trace import SpanAttributes

from openinference.instrumentation.claude_agent_sdk._wrappers import (
    _extract_model_name_from_usage,
    _extract_usage_and_cost_attributes,
    _TurnUsageAccumulator,
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


# ---------------------------------------------------------------------------
# `_extract_usage_and_cost_attributes` — cache tokens fold into prompt/total
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "usage,expected_prompt,expected_completion,expected_total,"
    "expected_cache_read,expected_cache_write",
    [
        pytest.param(
            # Payload from the package's own cassette.
            {
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_read_input_tokens": 17024,
                "cache_creation_input_tokens": 0,
            },
            17027,
            4,
            17031,
            17024,
            0,
            id="cache_read_folded_into_prompt_and_total",
        ),
        pytest.param(
            # Both cache terms nonzero catches a dropped term.
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 500,
            },
            710,
            5,
            715,
            200,
            500,
            id="both_cache_terms_folded_into_prompt_and_total",
        ),
        pytest.param(
            {"input_tokens": 12, "output_tokens": 8},
            12,
            8,
            20,
            None,
            None,
            id="no_cache_tokens_leaves_prompt_and_total_unchanged",
        ),
        pytest.param(
            # input_tokens absent: prompt still emitted from cache tokens.
            {"output_tokens": 4, "cache_read_input_tokens": 17024},
            17024,
            4,
            17028,
            17024,
            None,
            id="cache_only_payload_still_yields_prompt_and_total",
        ),
        pytest.param(
            # output_tokens absent (error/aborted results): total = prompt.
            {"input_tokens": 3, "cache_read_input_tokens": 17024},
            17027,
            None,
            17027,
            17024,
            None,
            id="missing_output_tokens_still_yields_total",
        ),
        pytest.param(
            # Unparseable cache_write falls back to cache_creation.
            {
                "input_tokens": 3,
                "output_tokens": 4,
                "cache_write_input_tokens": {"ephemeral_5m_input_tokens": 500},
                "cache_creation_input_tokens": 500,
            },
            503,
            4,
            507,
            None,
            500,
            id="unparseable_cache_write_falls_back_to_cache_creation",
        ),
        pytest.param(
            {},
            None,
            None,
            None,
            None,
            None,
            id="empty_usage_sets_no_token_attributes",
        ),
    ],
)
def test_token_counts_fold_cache_tokens(
    usage: dict[str, Any],
    expected_prompt: int | None,
    expected_completion: int | None,
    expected_total: int | None,
    expected_cache_read: int | None,
    expected_cache_write: int | None,
) -> None:
    attrs = _extract_usage_and_cost_attributes({"usage": usage})
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == expected_prompt
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == expected_completion
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == expected_total
    assert (
        attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == expected_cache_read
    )
    assert (
        attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) == expected_cache_write
    )


# ---------------------------------------------------------------------------
# `modelUsage` on a result message covers the whole session, so it must not
# override the model an assistant message already named for the span.
# ---------------------------------------------------------------------------


def _model_name_after_turn(
    assistant_model: str | None, model_usage: dict[str, Any], init_model: str | None = None
) -> str | None:
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from openinference.instrumentation.claude_agent_sdk._wrappers import (
        _process_message,
        _TurnUsageAccumulator,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    span = provider.get_tracer("test").start_span("turn")
    turn_usage = _TurnUsageAccumulator()
    if init_model is not None:
        init_message = {"type": "system", "subtype": "init", "data": {"model": init_model}}
        turn_usage.add(init_message)
        _process_message(init_message, span)
    if assistant_model is not None:
        assistant = AssistantMessage(content=[TextBlock(text="hi")], model=assistant_model)
        turn_usage.add(assistant)
        _process_message(assistant, span)
    result = ResultMessage(
        subtype="success",
        duration_ms=1,
        duration_api_ms=1,
        is_error=False,
        num_turns=1,
        session_id="s",
        total_cost_usd=0.1,
        usage={"input_tokens": 10, "output_tokens": 20},
        result="ok",
        model_usage=model_usage,
    )
    turn_usage.add(result)
    _process_message(result, span, use_model_usage=not turn_usage.has_assistant_model)
    span.end()
    value = span.attributes.get(SpanAttributes.LLM_MODEL_NAME) if span.attributes else None
    return str(value) if value is not None else None


def test_result_model_usage_does_not_override_the_assistant_model() -> None:
    # Turn 2 after set_model(): usage still holds the earlier, heavier model.
    usage = {
        "claude-sonnet-5-5": {"outputTokens": 400},
        "claude-haiku-4-5-20251001": {"outputTokens": 60},
    }
    assert _model_name_after_turn("claude-haiku-4-5-20251001", usage) == "claude-haiku-4-5-20251001"


def test_result_model_usage_is_still_used_when_no_model_was_seen() -> None:
    usage = {
        "claude-haiku-4-5": {"outputTokens": 5},
        "claude-sonnet-5-5": {"outputTokens": 400},
    }
    assert _model_name_after_turn(None, usage) == "claude-sonnet-5-5"


def test_result_model_usage_overrides_a_model_named_only_by_init() -> None:
    # The init metadata model is not an assistant message; session modelUsage wins.
    usage = {
        "claude-haiku-4-5": {"outputTokens": 10},
        "claude-sonnet-5-5": {"outputTokens": 400},
    }
    assert _model_name_after_turn(None, usage, init_model="claude-haiku-4-5") == "claude-sonnet-5-5"


def test_turn_usage_accumulator_sums_distinct_root_assistant_messages() -> None:
    accumulator = _TurnUsageAccumulator()
    accumulator.add(
        {
            "type": "assistant",
            "message_id": "message-1",
            "parent_tool_use_id": None,
            "usage": {"input_tokens": 10, "output_tokens": 3, "cache_read_input_tokens": 5},
        }
    )
    accumulator.add(
        {
            "type": "assistant",
            "message_id": "message-2",
            "parent_tool_use_id": None,
            "usage": {"input_tokens": 20, "output_tokens": 4, "cache_creation_input_tokens": 7},
        }
    )
    assert accumulator.attributes() == {
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 42,
        SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 7,
        SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 49,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 5,
        SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 7,
    }


def test_turn_usage_accumulator_deduplicates_partial_message_snapshots() -> None:
    accumulator = _TurnUsageAccumulator()
    accumulator.add(
        {
            "message_id": "message-1",
            "parent_tool_use_id": None,
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    )
    accumulator.add(
        {
            "message_id": "message-1",
            "parent_tool_use_id": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )
    assert accumulator.attributes()[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 15


def test_turn_usage_accumulator_excludes_subagent_and_unidentified_messages() -> None:
    accumulator = _TurnUsageAccumulator()
    accumulator.add(
        {
            "message_id": "subagent-message",
            "parent_tool_use_id": "task-1",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
    )
    accumulator.add({"usage": {"input_tokens": 7, "output_tokens": 2}})
    assert accumulator.attributes() == {}
