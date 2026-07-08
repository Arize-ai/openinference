from typing import Any

import pytest
from google.genai import types

from openinference.instrumentation.google_adk._wrappers import _get_attributes_from_usage_metadata


@pytest.mark.parametrize(
    "usage_metadata, expected",
    [
        pytest.param(
            types.GenerateContentResponseUsageMetadata(
                total_token_count=110,
                prompt_token_count=10,
                prompt_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=7),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=3),
                ],
                candidates_token_count=80,
                candidates_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=11),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=69),
                ],
                thoughts_token_count=20,
            ),
            {
                "llm.token_count.total": 110,
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 100,
                "llm.token_count.completion_details.reasoning": 20,
                "llm.token_count.prompt_details.audio": 7,
                "llm.token_count.completion_details.audio": 11,
            },
            id="all_fields_vertex_style_not_inclusive",
        ),
        pytest.param(
            # For Gemini Developer API, candidates already includes thoughts.
            types.GenerateContentResponseUsageMetadata(
                total_token_count=376,
                prompt_token_count=46,
                candidates_token_count=330,
                thoughts_token_count=195,
            ),
            {
                "llm.token_count.total": 376,
                "llm.token_count.prompt": 46,
                "llm.token_count.completion": 330,
                "llm.token_count.completion_details.reasoning": 195,
            },
            id="gemini_api_style_inclusive_no_double_count",
        ),
        pytest.param(
            # For Vertex AI and ADK's LiteLLM, thoughts are reported separately and are not part of candidates.
            types.GenerateContentResponseUsageMetadata(
                total_token_count=1707,
                prompt_token_count=6,
                candidates_token_count=569,
                thoughts_token_count=1132,
            ),
            {
                "llm.token_count.total": 1707,
                "llm.token_count.prompt": 6,
                "llm.token_count.completion": 569 + 1132,
                "llm.token_count.completion_details.reasoning": 1132,
            },
            id="vertex_ai_style_not_inclusive_must_sum",
        ),
        pytest.param(
            # For non-reasoning model, thoughts are absent entirely so no reasoning attribute.
            types.GenerateContentResponseUsageMetadata(
                total_token_count=90,
                prompt_token_count=10,
                candidates_token_count=80,
            ),
            {
                "llm.token_count.total": 90,
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 80,
            },
            id="no_thoughts_token_count",
        ),
        pytest.param(
            # Do not under-report completion tokens when prompt tokens are missing.
            types.GenerateContentResponseUsageMetadata(
                total_token_count=110,
                candidates_token_count=80,
                thoughts_token_count=20,
            ),
            {
                "llm.token_count.total": 110,
                "llm.token_count.completion": 100,
                "llm.token_count.completion_details.reasoning": 20,
            },
            id="missing_prompt_token_count_defaults_to_additive",
        ),
        pytest.param(
            # Do not under-report completion tokens when total tokens are missing.
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=10,
                candidates_token_count=80,
                thoughts_token_count=20,
            ),
            {
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 100,
                "llm.token_count.completion_details.reasoning": 20,
            },
            id="missing_total_token_count_defaults_to_additive",
        ),
        pytest.param(
            # Do not under-report completion tokens when candidates tokens are missing.
            types.GenerateContentResponseUsageMetadata(
                total_token_count=20,
                prompt_token_count=0,
                thoughts_token_count=20,
            ),
            {
                "llm.token_count.total": 20,
                "llm.token_count.completion": 20,
                "llm.token_count.completion_details.reasoning": 20,
            },
            id="missing_candidates_token_count_prompt_zero_inclusive",
        ),
        pytest.param(
            # When candidates are explicitly 0 with a nonzero prompt then thoughts get added.
            types.GenerateContentResponseUsageMetadata(
                total_token_count=25,
                prompt_token_count=5,
                candidates_token_count=0,
                thoughts_token_count=20,
            ),
            {
                "llm.token_count.total": 25,
                "llm.token_count.prompt": 5,
                "llm.token_count.completion": 20,
                "llm.token_count.completion_details.reasoning": 20,
            },
            id="candidates_token_count_explicit_zero_not_inclusive",
        ),
        pytest.param(
            # When thoughts are explicitly 0 then no reasoning attribute should be emitted.
            types.GenerateContentResponseUsageMetadata(
                total_token_count=90,
                prompt_token_count=10,
                candidates_token_count=80,
                thoughts_token_count=0,
            ),
            {
                "llm.token_count.total": 90,
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 80,
            },
            id="thoughts_token_count_explicit_zero_no_reasoning_detail",
        ),
    ],
)
def test_get_attributes_from_usage_metadata(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
    expected: dict[str, Any],
) -> None:
    actual = dict(_get_attributes_from_usage_metadata(usage_metadata))
    assert actual == expected
