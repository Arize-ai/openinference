import json
from typing import Any, Dict, List, Optional

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openinference.semconv.trace import SpanAttributes

from openinference.instrumentation.openai._response_accumulator import _ChatCompletionAccumulator


def _chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    choice: Dict[str, Any] = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return ChatCompletionChunk.model_validate(
        {
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o",
            "choices": [choice],
        }
    )


def _accumulated_message(chunks: List[ChatCompletionChunk]) -> Dict[str, Any]:
    accumulator = _ChatCompletionAccumulator(
        request_parameters={}, chat_completion_type=ChatCompletion
    )
    for chunk in chunks:
        accumulator.process_chunk(chunk)
    output = dict(accumulator.get_attributes())[SpanAttributes.OUTPUT_VALUE]
    assert isinstance(output, str)
    message: Dict[str, Any] = json.loads(output)["choices"][0]["message"]
    return message


@pytest.mark.parametrize("field", ["content", "refusal", "reasoning_content"])
def test_streamed_string_fields_are_concatenated(field: str) -> None:
    chunks = [
        _chunk({"role": "assistant", field: "I can"}),
        _chunk({field: "not help"}),
        _chunk({field: " with that."}, finish_reason="stop"),
    ]

    message = _accumulated_message(chunks)

    assert message[field] == "I cannot help with that."
    assert message["role"] == "assistant"


def test_reasoning_and_content_accumulate_independently() -> None:
    chunks = [
        _chunk({"role": "assistant", "reasoning_content": "First, "}),
        _chunk({"reasoning_content": "think."}),
        _chunk({"content": "The answer"}),
        _chunk({"content": " is 4."}, finish_reason="stop"),
    ]

    message = _accumulated_message(chunks)

    assert message["reasoning_content"] == "First, think."
    assert message["content"] == "The answer is 4."
