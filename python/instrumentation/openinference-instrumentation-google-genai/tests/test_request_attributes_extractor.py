from typing import Any

import pytest
from google.genai import types

from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)


@pytest.mark.parametrize(
    "system_instruction, expected",
    [
        # Verify plain text
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
        # Verify Content (SDK object)
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
            "",
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
        # Verify ContentDict
        pytest.param(
            {"parts": [{"text": "You are a helpful assistant."}]},
            "You are a helpful assistant.",
            id="content_dict_single_text_part",
        ),
        pytest.param(
            {
                "role": "user",
                "parts": [
                    {"text": "First instruction."},
                    {"text": "Second instruction."},
                ],
            },
            "First instruction.\n\nSecond instruction.",
            id="content_dict_with_role_multiple_parts",
        ),
        pytest.param(
            {"parts": []},
            "",
            id="content_dict_empty_parts",
        ),
        pytest.param(
            {
                "parts": [
                    {"inline_data": {"mime_type": "audio/mp3", "data": ""}},
                    {"text": "Text after non-text part."},
                ]
            },
            "Text after non-text part.",
            id="content_dict_skips_non_text_parts",
        ),
        # Verify Part (SDK object)
        pytest.param(
            types.Part(text="Bare part prompt."),
            "Bare part prompt.",
            id="bare_part_with_text",
        ),
        pytest.param(
            types.Part(inline_data=types.Blob(mime_type="audio/mp3", data=b"")),
            "",
            id="bare_part_non_text_yields_empty",
        ),
        # Verify PartDict
        pytest.param(
            {"text": "Bare part dict prompt."},
            "Bare part dict prompt.",
            id="part_dict_with_text",
        ),
        pytest.param(
            {"inline_data": {"mime_type": "audio/mp3", "data": ""}},
            "",
            id="part_dict_non_text_yields_empty",
        ),
        # Verify list[PartUnion] (SDK objects)
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
            id="list_mixed_sdk_types",
        ),
        # Verify list[PartUnionDict]
        pytest.param(
            [
                {"parts": [{"text": "Block one."}]},
                {"parts": [{"text": "Block two."}]},
            ],
            "Block one.\n\nBlock two.",
            id="list_of_content_dicts",
        ),
        pytest.param(
            [
                "Plain string.",
                {"parts": [{"text": "Content dict block."}]},
                {"text": "Part dict."},
            ],
            "Plain string.\n\nContent dict block.\n\nPart dict.",
            id="list_mixed_dict_types",
        ),
        pytest.param(
            [],
            "",
            id="empty_list",
        ),
        # Verify None
        pytest.param(
            None,
            "",
            id="none_yields_empty_string",
        ),
    ],
)
def test_normalize_system_instruction(
    system_instruction: Any,
    expected: str | None,
) -> None:
    result = _RequestAttributesExtractor()._normalize_system_instruction(system_instruction)
    assert isinstance(result, str), (
        f"Return value must always be a plain str for OTel compatibility, got {type(result)}"
    )
    if expected is not None:
        assert result == expected
