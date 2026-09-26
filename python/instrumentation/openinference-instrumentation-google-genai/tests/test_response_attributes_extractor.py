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


@pytest.mark.parametrize(
    "usage_metadata, expected",
    [
        pytest.param(
            types.GenerateContentResponseUsageMetadata(
                cached_content_token_count=20,
                cache_tokens_details=[
                    types.ModalityTokenCount(modality=types.MediaModality.AUDIO, token_count=14),
                    types.ModalityTokenCount(modality=types.MediaModality.TEXT, token_count=6),
                ],
                total_token_count=110,
                prompt_token_count=30,
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
                "llm.token_count.total": 130,
                "llm.token_count.prompt": 30,
                "llm.token_count.completion": 100,
                "llm.token_count.completion_details.reasoning": 20,
                "llm.token_count.prompt_details.cache_read": 20,
                "llm.token_count.prompt_details.audio": 7,
                "llm.token_count.completion_details.audio": 11,
            },
            id="all_fields",
        ),
    ],
)
def test_get_attributes_from_generate_content_usage_cached(
    usage_metadata: types.GenerateContentResponseUsageMetadata,
    expected: dict[str, Any],
) -> None:
    actual = dict(
        _ResponseAttributesExtractor()._get_attributes_from_generate_content_usage(usage_metadata)
    )
    assert actual == expected

def test_automatic_function_calling_history_is_output_message() -> None:
    first_function_call = types.FunctionCall(
        name="get_weather",
        args={"city": "Chennai"},
    )
    second_function_call = types.FunctionCall(
        name="get_time",
        args={"city": "Chennai"},
    )
    third_function_call = types.FunctionCall(
        name="get_temperature",
        args={"city": "Chennai"},
    )

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="I will check that for you."),
                    ],
                ),
            )
        ],
        automatic_function_calling_history=[
            types.Content(
                role="user",
                parts=[types.Part(text="What is the weather?")],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part(function_call=first_function_call),
                    types.Part(function_call=second_function_call),
                ],
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part(function_call=third_function_call),
                ],
            ),
        ],
    )

    actual = dict(
        _ResponseAttributesExtractor().get_attributes(
            response,
            {"contents": ["What is the weather?"]},
        )
    )

    assert (
        actual[
            "llm.output_messages.1.message.tool_calls.0.tool_call.function.name"
        ]
        == "get_weather"
    )
    assert (
        actual[
            "llm.output_messages.1.message.tool_calls.1.tool_call.function.name"
        ]
        == "get_time"
    )
    assert (
        actual[
            "llm.output_messages.2.message.tool_calls.0.tool_call.function.name"
        ]
        == "get_temperature"
    )

    assert "llm.output_messages.1.message.role" in actual
    assert "llm.output_messages.2.message.role" in actual

    assert not any(
        key.startswith("message.tool_calls.")
        for key in actual
    )


def test_afc_history_entry_that_is_the_final_candidate_is_not_published_twice() -> None:
    """The same Content object must not be published twice."""
    call_content = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    name="get_weather",
                    args={"location": "San Francisco"},
                )
            )
        ],
    )

    response = types.GenerateContentResponse(
        candidates=[types.Candidate(index=0, content=call_content)],
        automatic_function_calling_history=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="weather?")],
            ),
            call_content,
        ],
    )

    attributes = dict(
        _ResponseAttributesExtractor().get_attributes(response, {})
    )

    tool_call_keys = [key for key in attributes if "tool_calls" in key]
    duplicate_bucket = "llm.output_messages.1."

    assert [key for key in tool_call_keys if key.startswith(duplicate_bucket)] == []
    assert (
        attributes[
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name"
        ]
        == "get_weather"
    )


def test_afc_history_seeded_with_the_request_does_not_leak_earlier_turns() -> None:
    """Request-seeded history belongs to input messages, not this span's output."""
    earlier_call = types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(
                    name="get_weather",
                    args={"location": "San Francisco"},
                )
            )
        ],
    )

    request_contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="weather?")],
        ),
        earlier_call,
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        name="get_weather",
                        response={"temperature": 65},
                    )
                )
            ],
        ),
    ]

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="72 in New York")],
                ),
            )
        ],
        automatic_function_calling_history=[
            *request_contents,
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="get_weather",
                            args={"location": "NY"},
                        )
                    )
                ],
            ),
        ],
    )

    attributes = dict(
        _ResponseAttributesExtractor().get_attributes(
            response,
            {"contents": request_contents},
        )
    )

    output_tool_call = (
        "llm.output_messages.1."
        "message.tool_calls.0.tool_call.function.arguments"
    )

    assert attributes[output_tool_call] == '{"location": "NY"}'
    assert not any(
        "San Francisco" in str(value)
        and key.startswith("llm.output_messages")
        for key, value in attributes.items()
    )


def test_afc_history_after_merged_request_parts_keeps_this_call_turn() -> None:
    """The seeded prefix must use the SDK's transformed content count."""
    request_parts = [
        types.Part.from_text(text="What's the weather like"),
        types.Part.from_text(text="in San Francisco?"),
    ]

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                index=0,
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text="65 degrees and foggy.")],
                ),
            )
        ],
        automatic_function_calling_history=[
            types.Content(
                role="user",
                parts=request_parts,
            ),
            types.Content(
                role="model",
                parts=[
                    types.Part(
                        function_call=types.FunctionCall(
                            name="get_weather",
                            args={"location": "San Francisco"},
                        )
                    )
                ],
            ),
        ],
    )

    attributes = dict(
        _ResponseAttributesExtractor().get_attributes(
            response,
            {"contents": request_parts},
        )
    )

    tool_call_prefix = (
        "llm.output_messages.1."
        "message.tool_calls.0."
    )

    assert (
        attributes[f"{tool_call_prefix}tool_call.function.name"]
        == "get_weather"
    )
