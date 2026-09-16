from typing import Any, Dict

import respx
from httpx import Response
from mistralai.client import Mistral
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

REASONING = "The user asks for the capital of France. It is Paris."
ANSWER = "Paris."

# The shape magistral models return: the assistant content is a list of chunks,
# a thinking chunk (itself a list of text chunks) followed by the answer.
MAGISTRAL_RESPONSE = {
    "id": "b3f1d1f0e8d94f8f9d4b8d7e5a1c2b3d",
    "object": "chat.completion",
    "created": 1758000000,
    "model": "magistral-medium-latest",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": [{"type": "text", "text": REASONING}]},
                    {"type": "text", "text": ANSWER},
                ],
                "tool_calls": None,
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 20, "total_tokens": 61, "completion_tokens": 41},
}


@respx.mock
def test_reasoning_content_chunks_are_emitted_as_content_blocks(
    mistral_sync_client: Mistral,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    in_memory_span_exporter.clear()
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(200, json=MAGISTRAL_RESPONSE)
    )
    mistral_sync_client.chat.complete(
        model="magistral-medium-latest",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})

    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    contents = f"{output}.{MessageAttributes.MESSAGE_CONTENTS}"
    assert attributes[f"{output}.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert (
        attributes[f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "reasoning"
    )
    assert attributes[f"{contents}.0.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == REASONING
    assert attributes[f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"] == "text"
    assert attributes[f"{contents}.1.{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"] == ANSWER
    # A list of chunks is not a valid attribute value, so it must not be set as the plain content.
    assert f"{output}.{MessageAttributes.MESSAGE_CONTENT}" not in attributes
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 41


@respx.mock
def test_plain_string_content_is_unchanged(
    mistral_sync_client: Mistral,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    in_memory_span_exporter.clear()
    response = dict(MAGISTRAL_RESPONSE)
    response["choices"] = [
        {
            "index": 0,
            "message": {"role": "assistant", "content": ANSWER, "tool_calls": None},
            "finish_reason": "stop",
        }
    ]
    respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=Response(200, json=response)
    )
    mistral_sync_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})
    output = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes[f"{output}.{MessageAttributes.MESSAGE_CONTENT}"] == ANSWER
    assert not any(MessageAttributes.MESSAGE_CONTENTS in key for key in attributes)
