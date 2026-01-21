from typing import Any

import pytest
from google.genai.types import Content, Part

from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)


class TestContentToString:
    """Tests for _RequestAttributesExtractor._content_to_string method."""

    def test_string_input_returns_unchanged(self) -> None:
        """String input should be returned as-is."""
        extractor = _RequestAttributesExtractor()
        result = extractor._content_to_string("Hello, world!")
        assert result == "Hello, world!"

    def test_content_with_single_text_part(self) -> None:
        """Content with a single text part should extract the text."""
        extractor = _RequestAttributesExtractor()
        content = Content(
            role="user",
            parts=[Part.from_text(text="You are a helpful assistant.")],
        )
        result = extractor._content_to_string(content)
        assert result == "You are a helpful assistant."

    def test_content_with_multiple_text_parts(self) -> None:
        """Content with multiple text parts should join them with double newlines."""
        extractor = _RequestAttributesExtractor()
        content = Content(
            role="user",
            parts=[
                Part.from_text(text="First instruction."),
                Part.from_text(text="Second instruction."),
            ],
        )
        result = extractor._content_to_string(content)
        assert result == "First instruction.\n\nSecond instruction."

    def test_content_with_empty_parts(self) -> None:
        """Content with no parts should fall back to str representation."""
        extractor = _RequestAttributesExtractor()
        content = Content(role="user", parts=[])
        result = extractor._content_to_string(content)
        # Should fall back to str(content) since there are no text parts
        assert isinstance(result, str)

    def test_fallback_for_other_types(self) -> None:
        """Non-string, non-Content types should be converted via str()."""
        extractor = _RequestAttributesExtractor()
        result = extractor._content_to_string(12345)
        assert result == "12345"

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            pytest.param(
                "Simple string instruction",
                "Simple string instruction",
                id="simple_string",
            ),
            pytest.param(
                Content(
                    role="system",
                    parts=[Part.from_text(text="Be helpful and concise.")],
                ),
                "Be helpful and concise.",
                id="content_single_part",
            ),
            pytest.param(
                Content(
                    role="system",
                    parts=[
                        Part.from_text(text="Be helpful."),
                        Part.from_text(text="Be concise."),
                        Part.from_text(text="Be accurate."),
                    ],
                ),
                "Be helpful.\n\nBe concise.\n\nBe accurate.",
                id="content_multiple_parts",
            ),
        ],
    )
    def test_content_to_string_parametrized(
        self, input_value: Any, expected: str
    ) -> None:
        """Parametrized tests for various input types."""
        extractor = _RequestAttributesExtractor()
        result = extractor._content_to_string(input_value)
        assert result == expected
