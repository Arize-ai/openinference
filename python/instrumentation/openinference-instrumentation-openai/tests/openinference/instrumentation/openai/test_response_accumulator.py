import pytest
from openai.types.chat import ChatCompletionChunk
from openinference.instrumentation.openai._response_accumulator import (
    _ChatCompletionAccumulator,
)


def test_chat_completion_accumulator(chat_completion_chunks, desired_chat_completion_result):
    accumulator = _ChatCompletionAccumulator({})
    for chunk in chat_completion_chunks:
        accumulator.process_chunk(chunk)
    assert accumulator._result() == desired_chat_completion_result


@pytest.fixture
def desired_chat_completion_result():
    return {
        "id": "xyz",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"content": "A1", "role": "assistant"},
            },
            {
                "index": 1,
                "finish_reason": "stop",
                "message": {"content": "B2", "role": "assistant"},
            },
            {
                "index": 2,
            },
            {
                "index": 3,
                "message": {
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": "C3"}},
                        {"index": 1, "function": {"arguments": "D4"}},
                    ]
                },
            },
        ],
        "created": 123,
        "model": "ultra-turbo",
    }


@pytest.fixture
def chat_completion_chunks():
    chunks = [
        {
            "id": "xyz",
            "choices": [{"index": 0, "delta": {"content": "A"}}],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [
                {
                    "index": 3,
                    "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "D"}}]},
                },
            ],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [{"index": 1, "delta": {"content": "B"}, "finish_reason": None}],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [
                {
                    "index": 3,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "C"}}]},
                },
            ],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [{"index": 1, "delta": {"content": "2"}}],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [{"index": 0, "delta": {"content": "1"}, "finish_reason": None}],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [{"index": 1, "delta": {"role": "assistant"}}],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [{"finish_reason": "length", "index": 0}],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [
                {
                    "index": 3,
                    "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "4"}}]},
                },
            ],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [{"finish_reason": "stop", "index": 1}],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [
                {
                    "index": 3,
                    "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "3"}}]},
                },
            ],
            "created": 123,
            "model": "ultra-turbo",
        },
        {
            "id": "xyz",
            "choices": [{"index": 2, "delta": {"content": ""}}],
            "created": 123,
            "model": "ultra-turbo",
        },
    ]
    return [ChatCompletionChunk.construct(**chunk) for chunk in chunks]
