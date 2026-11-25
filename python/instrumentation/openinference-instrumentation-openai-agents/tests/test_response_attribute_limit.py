"""Regression test for the OTel 128-attribute limit on response spans.

``ResponseSpanData`` spans carry per-message input attributes as well as response
attributes (tools, token counts, output, model name). The SDK's attribute store keeps
only the 128 most recently set attributes and evicts the oldest ones, so the order in
which the processor sets them decides which half of the data survives once a
conversation grows long enough. Response-side attributes must be set last: they are
small in number but are the ones consumers depend on for token accounting.
"""

from __future__ import annotations

from typing import Any

import pytest
from agents.tracing.span_data import ResponseSpanData
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputItemParam,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseUsage,
)
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig
from openinference.instrumentation.openai_agents._processor import OpenInferenceTracingProcessor

_TRACE_ID = "trace-attribute-limit"


class _FakeTrace:
    trace_id = _TRACE_ID
    name = "Agent workflow"


class _FakeSpan:
    span_id = "response"
    parent_id = None
    started_at = "2020-01-01T00:00:00+00:00"
    ended_at = "2020-01-01T00:00:01+00:00"
    error: Any = None

    def __init__(self, span_data: ResponseSpanData) -> None:
        self.span_data = span_data
        self.trace_id = _TRACE_ID


def _response_with_usage() -> Response:
    """A minimal real Response: the isinstance check in the processor requires one."""
    usage = ResponseUsage(
        input_tokens=123,
        input_tokens_details=InputTokensDetails(cached_tokens=7),
        output_tokens=45,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=3),
        total_tokens=168,
    )
    return Response(
        id="resp-usage",
        created_at=0.0,
        model="gpt-4o-mini",
        object="response",
        output=[
            ResponseOutputMessage(
                id="m1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Done.", annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
    )


def _run_response_span(message_count: int) -> ReadableSpan:
    """Drive one response span through a real TracerProvider with default limits."""
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer(__name__), config=TraceConfig())
    processor = OpenInferenceTracingProcessor(tracer)  # type: ignore[arg-type]

    newest = f"final message {message_count}"
    messages: list[ResponseInputItemParam] = [
        EasyInputMessageParam(role="user", content=f"message {i}") for i in range(1, message_count)
    ]
    messages.append(EasyInputMessageParam(role="user", content=newest))

    trace = _FakeTrace()
    span = _FakeSpan(ResponseSpanData(response=_response_with_usage(), input=messages))
    processor.on_trace_start(trace)  # type: ignore[arg-type]
    processor.on_span_start(span)  # type: ignore[arg-type]
    processor.on_span_end(span)  # type: ignore[arg-type]
    processor.on_trace_end(trace)  # type: ignore[arg-type]

    llm_spans = [
        s
        for s in exporter.get_finished_spans()
        if dict(s.attributes or {}).get("openinference.span.kind") == "LLM"
    ]
    assert len(llm_spans) == 1
    return llm_spans[0]


@pytest.mark.parametrize("message_count", [60, 80])
def test_response_attributes_survive_attribute_limit(message_count: int) -> None:
    span = _run_response_span(message_count)
    attrs = dict(span.attributes or {})

    # Each message yields two attributes, so the SDK's 128-attribute limit is exceeded
    # and the oldest input messages are evicted.
    assert span.dropped_attributes > 0

    # The response-side attributes, set last, must survive the eviction.
    assert attrs["llm.token_count.prompt"] == 123
    assert attrs["llm.token_count.completion"] == 45
    assert attrs["llm.token_count.total"] == 168
    assert attrs["llm.model_name"] == "gpt-4o-mini"
    assert "output.value" in attrs
    assert "Done." in str(attrs["output.value"])

    # The most recent input message, set after its predecessors, must survive too.
    assert attrs[f"llm.input_messages.{message_count}.message.content"] == (
        f"final message {message_count}"
    )
