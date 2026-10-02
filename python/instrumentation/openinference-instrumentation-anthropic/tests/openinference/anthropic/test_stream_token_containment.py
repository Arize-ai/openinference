"""Regression tests for #3499: token-count failures in the streaming extractor.

The non-streaming path wraps token-count extraction with @_stop_on_exception,
so a bad usage snapshot only loses the token counts. The streaming
_MessageExtractor must behave the same way: without containment, the
exception escapes into _finish_tracing, which exports the span with no
attributes at all (no model, no messages, no token counts).
"""

import logging

import pytest
from anthropic.types import Message, TextBlock, Usage

from openinference.instrumentation.anthropic._stream import _MessageExtractor
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)


def test_message_extractor_contains_token_count_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    usage = Usage(input_tokens=10, output_tokens=20)
    # Simulate the partial/delta-shaped usage from #3499: input_tokens arrives
    # as None, so _get_token_counts raises TypeError on `None + 0`.
    object.__setattr__(usage, "input_tokens", None)
    snapshot = Message(
        id="msg_stream_token_failure",
        content=[TextBlock(type="text", text="Paris.")],
        model="claude-opus-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=usage,
    )

    with caplog.at_level(logging.ERROR):
        attributes = dict(_MessageExtractor(snapshot).get_attributes())

    # The failure is logged, not raised.
    assert "Failed to get token counts from streaming snapshot." in caplog.text
    # Everything except the token counts survives.
    assert attributes[SpanAttributes.LLM_MODEL_NAME] == "claude-opus-4-6"
    output_message = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    assert attributes[f"{output_message}.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert (
        attributes[
            f"{output_message}.{MessageAttributes.MESSAGE_CONTENTS}.0."
            f"{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"
        ]
        == "Paris."
    )
    assert attributes[SpanAttributes.LLM_FINISH_REASON] == "end_turn"
    assert SpanAttributes.LLM_TOKEN_COUNT_PROMPT not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_COMPLETION not in attributes
    assert SpanAttributes.LLM_TOKEN_COUNT_TOTAL not in attributes
