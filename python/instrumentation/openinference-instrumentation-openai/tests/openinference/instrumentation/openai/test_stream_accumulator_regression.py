"""Regression tests for the OpenAI Chat Completions streaming accumulator.

These tests verify that the ``refusal`` and ``reasoning_content`` delta fields
emitted by ``ChatCompletionChunk`` are concatenated across stream chunks rather
than overwritten by the last fragment. They exercise the real ``_Stream``,
``_WithSpan``, and ``_ChatCompletionAccumulator`` types end-to-end and assert
the resulting ``SpanAttributes.OUTPUT_VALUE`` JSON produced by
``InMemorySpanExporter``.
"""

import json
from typing import List

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openai._response_accumulator import (
    _ChatCompletionAccumulator,
)
from openinference.instrumentation.openai._stream import _Stream
from openinference.instrumentation.openai._with_span import _WithSpan
from openinference.semconv.trace import SpanAttributes


def _build_stream_chunks(
    *,
    delta_field: str,
    fragments: List[str],
    finish_reason: str = "stop",
) -> List[ChatCompletionChunk]:
    """Build a realistic ``ChatCompletionChunk`` sequence.

    The sequence is: one role chunk, ``len(fragments)`` content chunks that
    emit ``delta_field`` fragments, and one final chunk with
    ``finish_reason=finish_reason``.
    """
    chunks: List[ChatCompletionChunk] = []
    chunks.append(
        ChatCompletionChunk.construct(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=0,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        )
    )
    for index, fragment in enumerate(fragments):
        chunks.append(
            ChatCompletionChunk.construct(
                id="chatcmpl-test",
                object="chat.completion.chunk",
                created=0,
                model="test-model",
                choices=[
                    {
                        "index": 0,
                        "delta": {delta_field: fragment},
                        "finish_reason": None,
                    }
                ],
            )
        )
    chunks.append(
        ChatCompletionChunk.construct(
            id="chatcmpl-test",
            object="chat.completion.chunk",
            created=0,
            model="test-model",
            choices=[
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": finish_reason,
                }
            ],
        )
    )
    return chunks


def _exhaust_stream(chunks: List[ChatCompletionChunk]) -> InMemorySpanExporter:
    """Build a fresh provider/exporter pair and drain ``chunks`` through ``_Stream``."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter=exporter))

    tracer = provider.get_tracer(__name__)
    span = tracer.start_span(__name__)
    with_span = _WithSpan(span=span)
    accumulator = _ChatCompletionAccumulator(
        request_parameters={},
        chat_completion_type=ChatCompletion,
    )

    wrapped_chunks = iter(chunks)
    stream = _Stream(
        stream=wrapped_chunks,  # type: ignore[arg-type]
        with_span=with_span,
        response_accumulator=accumulator,
    )
    for _ in stream:
        pass

    return exporter


def test_streamed_refusal_full_string() -> None:
    fragments = ["I can", "'t help", " with that."]
    exporter = _exhaust_stream(
        _build_stream_chunks(delta_field="refusal", fragments=fragments)
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    output_value = span.attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert isinstance(output_value, str)

    parsed = json.loads(output_value)
    assert parsed["choices"][0]["message"]["refusal"] == "".join(fragments)


def test_streamed_reasoning_content_full_string() -> None:
    fragments = ["I can", "'t help", " with that."]
    exporter = _exhaust_stream(
        _build_stream_chunks(delta_field="reasoning_content", fragments=fragments)
    )

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    output_value = span.attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert isinstance(output_value, str)

    parsed = json.loads(output_value)
    assert parsed["choices"][0]["message"]["reasoning_content"] == "".join(fragments)
