from types import SimpleNamespace

from openinference.instrumentation.google_genai.interactions_attributes import (
    get_attributes_from_response,
    get_token_object_from_response,
)
from openinference.semconv.trace import SpanAttributes


def test_interaction_usage_falls_back_to_input_tokens_by_modality() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            total_tokens=1853,
            total_input_tokens=0,
            input_tokens_by_modality=[
                SimpleNamespace(modality="text", tokens=84),
                SimpleNamespace(modality="image", tokens=1000),
            ],
            total_cached_tokens=0,
            total_output_tokens=600,
            total_tool_use_tokens=0,
            total_thought_tokens=169,
        )
    )

    assert get_token_object_from_response(response) == {
        "total": 1853,
        "prompt": 1084,
        "completion": 769,
    }


def test_interaction_usage_counts_tool_use_as_completion() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            total_tokens=97,
            total_input_tokens=85,
            input_tokens_by_modality=[SimpleNamespace(modality="text", tokens=85)],
            total_cached_tokens=0,
            total_output_tokens=0,
            total_tool_use_tokens=12,
            total_thought_tokens=0,
        )
    )

    assert get_token_object_from_response(response) == {
        "total": 97,
        "prompt": 85,
        "completion": 12,
    }


def test_interaction_usage_reports_cached_tokens_as_prompt_details() -> None:
    response = SimpleNamespace(
        usage=SimpleNamespace(
            total_tokens=100,
            total_input_tokens=50,
            input_tokens_by_modality=[],
            total_cached_tokens=20,
            total_output_tokens=30,
            total_tool_use_tokens=0,
            total_thought_tokens=0,
        )
    )

    assert get_token_object_from_response(response) == {
        "total": 100,
        "prompt": 70,
        "completion": 30,
        "prompt_details": {"cache_read": 20},
    }


def test_agent_response_sets_agent_span_kind_without_create_request() -> None:
    attributes = get_attributes_from_response(
        {},
        {
            "id": "interaction-id",
            "status": "in_progress",
            "agent": "deep-research-pro-preview-12-2025",
        },
    )

    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "AGENT"


def test_model_response_sets_llm_span_kind_without_create_request() -> None:
    attributes = get_attributes_from_response(
        {},
        {
            "model": "gemini-2.5-flash",
            "steps": [],
            "usage": {
                "total_tokens": 20,
                "total_input_tokens": 7,
                "input_tokens_by_modality": [{"modality": "text", "tokens": 7}],
                "total_cached_tokens": 0,
                "total_output_tokens": 13,
                "total_tool_use_tokens": 0,
                "total_thought_tokens": 0,
            },
        },
    )

    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "LLM"
    assert attributes[SpanAttributes.LLM_PROVIDER] == "google"
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 7
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 13
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 20
