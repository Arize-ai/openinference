from types import SimpleNamespace
from typing import Any, Optional

import pytest
from llama_index.core.callbacks import CBEventType, EventPayload
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMCompletionEndEvent,
    LLMCompletionInProgressEvent,
)
from llama_index.core.llms import ChatMessage, ChatResponse, CompletionResponse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import TracerProvider

from openinference.instrumentation.llama_index._callback import payload_to_semantic_attributes
from openinference.instrumentation.llama_index._finish_reason import _extract_finish_reason
from openinference.instrumentation.llama_index._handler import _Span
from openinference.semconv.trace import SpanAttributes


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"choices": [{"finish_reason": "tool_calls"}]}, "tool_calls"),
        ({"choices": [{"finish_reason": "stop"}, {"finish_reason": "length"}]}, "stop"),
        (SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")]), "stop"),
        ({"stop_reason": "end_turn"}, "end_turn"),
        (SimpleNamespace(stopReason="max_tokens"), "max_tokens"),
        ({"results": [{"completionReason": "FINISH"}]}, "FINISH"),
        ({"generations": [{"finish_reason": "length"}]}, "length"),
        ({"outputs": [{"stop_reason": "end"}]}, "end"),
        ({"completions": [{"finishReason": {"reason": "stop"}}]}, "stop"),
        ({"choices": [{"finish_reason": None}]}, None),
        ({"choices": []}, None),
        ({"finish_reason": 42}, None),
    ],
)
def test_extract_finish_reason(raw: Any, expected: Optional[str]) -> None:
    assert _extract_finish_reason(SimpleNamespace(raw=raw)) == expected


def test_extract_finish_reason_from_additional_kwargs() -> None:
    response = SimpleNamespace(raw={}, additional_kwargs={"stop_reason": "end_turn"})
    assert _extract_finish_reason(response) == "end_turn"


@pytest.mark.parametrize("response_type", ["chat", "completion"])
def test_legacy_callback_finish_reason(response_type: str) -> None:
    if response_type == "chat":
        response = ChatResponse(
            message=ChatMessage(content="hello"),
            raw={"stop_reason": "end_turn"},
        )
    else:
        response = CompletionResponse(text="hello", raw={"choices": [{"finish_reason": "stop"}]})  # type: ignore[assignment]

    attributes = payload_to_semantic_attributes(
        CBEventType.LLM,
        {EventPayload.RESPONSE: response},
    )
    assert attributes[SpanAttributes.LLM_FINISH_REASON] == (
        "end_turn" if response_type == "chat" else "stop"
    )


@pytest.mark.parametrize("response_type", ["chat", "completion"])
@pytest.mark.parametrize("finish_reason", ["stop", None])
def test_stream_finish_reason_survives_final_chunk_without_reason(
    response_type: str,
    finish_reason: Optional[str],
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    span = _Span(otel_span=tracer_provider.get_tracer(__name__).start_span("llm"), span_kind="LLM")
    raw = {"choices": [{"finish_reason": finish_reason}]} if finish_reason else {"choices": []}
    if response_type == "chat":
        message = ChatMessage(content="hello")
        span.process_event(
            LLMChatInProgressEvent(messages=[], response=ChatResponse(message=message, raw=raw))
        )
        span.process_event(
            LLMChatEndEvent(
                messages=[], response=ChatResponse(message=message, raw={"choices": []})
            )
        )
    else:
        span.process_event(
            LLMCompletionInProgressEvent(
                prompt="hi", response=CompletionResponse(text="hello", raw=raw)
            )
        )
        span.process_event(
            LLMCompletionEndEvent(
                prompt="hi", response=CompletionResponse(text="hello", raw={"choices": []})
            )
        )
    span.end()

    exported_span = in_memory_span_exporter.get_finished_spans()[0]
    assert exported_span.attributes is not None
    assert exported_span.attributes.get(SpanAttributes.LLM_FINISH_REASON) == finish_reason
