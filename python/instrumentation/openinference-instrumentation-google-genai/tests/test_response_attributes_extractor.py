from typing import Any

import pytest
from google.genai import types

from openinference.instrumentation.google_genai._response_attributes_extractor import (
    _ResponseAttributesExtractor,
)


@pytest.mark.parametrize(
    "obj, expected",
    [
        pytest.param(
            types.GenerateContentResponseUsageMetadata(
                total_token_count=110,
                prompt_token_count=10,
                candidates_token_count=80,
                thoughts_token_count=20,
            ),
            {
                "llm.token_count.total": 110,
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 100,
                "llm.token_count.completion_details.reasoning": 20,
            },
            id="all_fields",
        ),
    ],
)
def test_get_attributes_from_generate_content_usage(
    obj: types.GenerateContentResponseUsageMetadata,
    expected: dict[str, Any],
) -> None:
    actual = dict(_ResponseAttributesExtractor()._get_attributes_from_generate_content_usage(obj))
    assert actual == expected
