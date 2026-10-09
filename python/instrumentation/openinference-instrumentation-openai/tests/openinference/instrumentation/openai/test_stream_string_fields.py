import json
from typing import Any, Dict, Iterator, List, Optional

from openai.types.chat import ChatCompletionChunk
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openai._response_accumulator import _ChatCompletionAccumulator
from openinference.instrumentation.openai._stream import _Stream
from openinference.instrumentation.openai._with_span import _WithSpan


def _chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    choice: Dict[str, Any] = {"index": 0, "delta": delta}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    return ChatCompletionChunk(
        id="chatcmpl-x",
        object="chat.completion.chunk",
        created=1,
        model="gpt-4o",
        choices=[choice],
    )


def _output_value(chunks: List[ChatCompletionChunk]) -> Dict[str, Any]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    def _iter() -> Iterator[ChatCompletionChunk]:
        yield from chunks

    with tracer.start_as_current_span("ChatCompletion") as span:
        stream = _Stream(
            stream=_iter(),
            with_span=_WithSpan(span=span),
            response_accumulator=_ChatCompletionAccumulator(
                request_parameters={"model": "gpt-4o", "messages": []},
                chat_completion_type=ChatCompletionChunk,
                response_attributes_extractor=None,
            ),
        )
        for _ in stream:
            pass
    return json.loads(exporter.get_finished_spans()[0].attributes["output.value"])


def test_streamed_refusal_and_reasoning_content_are_concatenated() -> None:
    fragments = ["I can", "'t help", " with that."]
    expected = "".join(fragments)
    for field in ("refusal", "reasoning_content"):
        chunks = [_chunk({field: fragment}) for fragment in fragments]
        chunks.append(_chunk({"role": "assistant"}))
        chunks.append(_chunk({}, finish_reason="stop"))
        message = _output_value(chunks)["choices"][0]["message"]
        assert message.get(field) == expected


def test_stream_without_refusal_or_reasoning_content_omits_those_fields() -> None:
    chunks = [
        _chunk({"content": "hello"}),
        _chunk({"content": " world"}),
        _chunk({}, finish_reason="stop"),
    ]
    message = _output_value(chunks)["choices"][0]["message"]
    assert message.get("content") == "hello world"
    assert "refusal" not in message
    assert "reasoning_content" not in message
