from typing import Any

import pytest
from google.genai import types

from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)


@pytest.mark.parametrize(
    "content, expected",
    [
        pytest.param(
            "You are a helpful assistant.",
            "You are a helpful assistant.",
            id="plain_string",
        ),
        pytest.param(
            "",
            "",
            id="empty_string",
        ),
        pytest.param(
            types.Content(parts=[types.Part(text="You are a helpful assistant.")]),
            "You are a helpful assistant.",
            id="content_single_text_part",
        ),
        pytest.param(
            types.Content(
                parts=[
                    types.Part(text="First instruction."),
                    types.Part(text="Second instruction."),
                ]
            ),
            "First instruction.\n\nSecond instruction.",
            id="content_multiple_text_parts",
        ),
        pytest.param(
            types.Content(parts=[]),
            None,
            id="content_empty_parts",
        ),
        pytest.param(
            types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(mime_type="audio/mp3", data=b"")),
                    types.Part(text="Text after non-text part."),
                ]
            ),
            "Text after non-text part.",
            id="content_skips_non_text_parts",
        ),
        pytest.param(
            types.Part(text="Bare part prompt."),
            "Bare part prompt.",
            id="bare_part_with_text",
        ),
        pytest.param(
            [
                types.Content(parts=[types.Part(text="Block one.")]),
                types.Content(parts=[types.Part(text="Block two.")]),
            ],
            "Block one.\n\nBlock two.",
            id="list_of_content_objects",
        ),
        pytest.param(
            [
                "Plain string.",
                types.Content(parts=[types.Part(text="Content block.")]),
                types.Part(text="Bare part."),
            ],
            "Plain string.\n\nContent block.\n\nBare part.",
            id="list_mixed_types",
        ),
        pytest.param(
            [],
            "",
            id="empty_list",
        ),
    ],
)
def test_extract_text_from_content(
    content: Any,
    expected: str | None,
) -> None:
    result = _RequestAttributesExtractor()._extract_text_from_content(content)
    assert isinstance(result, str), (
        f"Return value must always be a plain str for OTel compatibility, got {type(result)}"
    )
    if expected is not None:
        assert result == expected
