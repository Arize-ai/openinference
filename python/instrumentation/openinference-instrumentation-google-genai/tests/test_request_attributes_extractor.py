from typing import Any, Dict

import pytest
from google.genai import types

from openinference.instrumentation.google_genai._request_attributes_extractor import (
    _RequestAttributesExtractor,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

_BASE_LLM_SPAN_ATTRS: dict[str, str] = {
    SpanAttributes.LLM_PROVIDER: "google",
    SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
}


def _im(msg_index: int, *parts: str) -> str:
    """Build `llm.input_messages.{msg_index}.*` attribute keys (segments joined with dots)."""
    return f"{SpanAttributes.LLM_INPUT_MESSAGES}.{msg_index}." + ".".join(parts)


@pytest.mark.parametrize(
    "system_instruction, expected",
    [
        pytest.param(
            "You are a helpful assistant.",
            {
                _im(0, MessageAttributes.MESSAGE_ROLE): "system",
                _im(0, MessageAttributes.MESSAGE_CONTENT): "You are a helpful assistant.",
            },
            id="plain_string",
        ),
        pytest.param(
            types.Content(parts=[types.Part(text="You are a helpful assistant.")]),
            {
                _im(0, MessageAttributes.MESSAGE_ROLE): "system",
                _im(0, MessageAttributes.MESSAGE_CONTENT): "You are a helpful assistant.",
            },
            id="content_single_text_part",
        ),
        pytest.param(
            types.Content(
                parts=[
                    types.Part(text="First instruction."),
                    types.Part(text="Second instruction."),
                ]
            ),
            {
                _im(0, MessageAttributes.MESSAGE_ROLE): "system",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "First instruction.",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Second instruction.",
            },
            id="content_multiple_text_parts",
        ),
        pytest.param(
            types.Content(
                parts=[
                    types.Part(inline_data=types.Blob(mime_type="audio/mp3", data=b"")),
                    types.Part(text="Text after non-text part."),
                ]
            ),
            {
                _im(0, MessageAttributes.MESSAGE_ROLE): "system",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Text after non-text part.",
            },
            id="content_skips_non_text_parts",
        ),
        pytest.param(
            types.Part(text="Bare part prompt."),
            {
                _im(0, MessageAttributes.MESSAGE_ROLE): "system",
                _im(0, MessageAttributes.MESSAGE_CONTENT): "Bare part prompt.",
            },
            id="bare_part_with_text",
        ),
        pytest.param(
            ["First instruction.", "Second instruction."],
            {
                _im(0, MessageAttributes.MESSAGE_ROLE): "system",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "First instruction.",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Second instruction.",
            },
            id="list_of_strings",
        ),
        pytest.param(
            [
                "Plain string.",
                types.Part(text="Bare part."),
            ],
            {
                _im(0, MessageAttributes.MESSAGE_ROLE): "system",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "0",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Plain string.",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TYPE,
                ): "text",
                _im(
                    0,
                    MessageAttributes.MESSAGE_CONTENTS,
                    "1",
                    MessageContentAttributes.MESSAGE_CONTENT_TEXT,
                ): "Bare part.",
            },
            id="list_mixed_string_and_part",
        ),
    ],
)
def test_system_instruction_extraction(
    system_instruction: Any,
    expected: Dict[str, Any],
) -> None:
    extractor = _RequestAttributesExtractor()
    request_parameters = {
        "config": types.GenerateContentConfig(system_instruction=system_instruction),
    }
    attrs = dict(extractor.get_attributes_from_request(request_parameters))
    assert attrs == {**_BASE_LLM_SPAN_ATTRS, **expected}


@pytest.mark.parametrize(
    "system_instruction",
    [
        pytest.param(None, id="none_yields_no_message"),
    ],
)
def test_system_instruction_falsy_input(system_instruction: Any) -> None:
    extractor = _RequestAttributesExtractor()
    request_parameters = {
        "config": types.GenerateContentConfig(system_instruction=system_instruction),
    }
    attrs = dict(extractor.get_attributes_from_request(request_parameters))
    assert attrs == _BASE_LLM_SPAN_ATTRS


def test_system_instruction_from_dict_config() -> None:
    """Config passed as a plain dict should be normalized via model_validate."""
    extractor = _RequestAttributesExtractor()
    request_parameters = {
        "config": {"system_instruction": "You are helpful."},
    }
    attrs = dict(extractor.get_attributes_from_request(request_parameters))
    role_key = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"
    content_key = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}"
    assert attrs == {
        **_BASE_LLM_SPAN_ATTRS,
        role_key: "system",
        content_key: "You are helpful.",
    }
