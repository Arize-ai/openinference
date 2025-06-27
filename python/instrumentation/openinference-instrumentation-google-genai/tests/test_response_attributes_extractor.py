from typing import Any

import pytest
from google.genai import types

from openinference.instrumentation.google_genai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)


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
            id="all_fields",
        ),
    ],
)
def test_get_attributes_from_generate_content_usage(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
    expected: dict[str, Any],
) -> None:
    actual = dict(
        _ResponseAttributesExtractor()._get_attributes_from_generate_content_usage(usage_metadata)
    )
    assert actual == expected
