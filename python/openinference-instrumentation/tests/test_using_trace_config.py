import asyncio
import json
from typing import Any, Dict

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import (
    REDACTED_VALUE,
    OITracer,
    TraceConfig,
    get_trace_config_from_context,
    using_trace_config,
)
from openinference.semconv.trace import SpanAttributes

HIDE_ALL = TraceConfig(hide_inputs=True, hide_outputs=True)


def _emit(tracer: OITracer, name: str = "span") -> None:
    with tracer.start_as_current_span(name, openinference_span_kind="chain") as span:
        span.set_input("secret-in")
        span.set_output("secret-out")


def _attrs(exporter: InMemorySpanExporter) -> Dict[str, Any]:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    return dict(spans[0].attributes or {})


def test_default_config_used_without_override(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    assert get_trace_config_from_context() is None
    _emit(tracer)
    attrs = _attrs(in_memory_span_exporter)
    assert attrs[SpanAttributes.INPUT_VALUE] == "secret-in"
    assert attrs[SpanAttributes.OUTPUT_VALUE] == "secret-out"


def test_override_applies_inside_block_only(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_trace_config(HIDE_ALL):
        assert get_trace_config_from_context() is HIDE_ALL
        _emit(tracer)
    attrs = _attrs(in_memory_span_exporter)
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE

    in_memory_span_exporter.clear()
    assert get_trace_config_from_context() is None
    _emit(tracer)
    assert _attrs(in_memory_span_exporter)[SpanAttributes.INPUT_VALUE] == "secret-in"


def test_nesting_restores_outer_config(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    outer = TraceConfig(hide_inputs=True)
    inner = TraceConfig(hide_outputs=True)
    with using_trace_config(outer):
        with using_trace_config(inner):
            assert get_trace_config_from_context() is inner
            _emit(tracer, "inner")
        assert get_trace_config_from_context() is outer
        _emit(tracer, "outer")
    spans = {s.name: dict(s.attributes or {}) for s in in_memory_span_exporter.get_finished_spans()}
    assert spans["inner"][SpanAttributes.INPUT_VALUE] == "secret-in"
    assert spans["inner"][SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert spans["outer"][SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert spans["outer"][SpanAttributes.OUTPUT_VALUE] == "secret-out"


def test_decorator_sync(tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter) -> None:
    @using_trace_config(HIDE_ALL)
    def fn() -> None:
        _emit(tracer)

    fn()
    assert _attrs(in_memory_span_exporter)[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert get_trace_config_from_context() is None


async def test_async_context_manager_and_decorator(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    async with using_trace_config(HIDE_ALL):
        await asyncio.sleep(0)
        _emit(tracer, "cm")

    @using_trace_config(HIDE_ALL)
    async def fn() -> None:
        await asyncio.sleep(0)
        _emit(tracer, "deco")

    await fn()
    spans = {s.name: dict(s.attributes or {}) for s in in_memory_span_exporter.get_finished_spans()}
    for name in ("cm", "deco"):
        assert spans[name][SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
        assert spans[name][SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    assert get_trace_config_from_context() is None


async def test_concurrent_tasks_are_isolated(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    async def hidden() -> None:
        with using_trace_config(HIDE_ALL):
            await asyncio.sleep(0)
            _emit(tracer, "hidden")

    async def visible() -> None:
        await asyncio.sleep(0)
        _emit(tracer, "visible")

    await asyncio.gather(hidden(), visible())
    spans = {s.name: dict(s.attributes or {}) for s in in_memory_span_exporter.get_finished_spans()}
    assert spans["hidden"][SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert spans["visible"][SpanAttributes.INPUT_VALUE] == "secret-in"


def test_span_ended_outside_block_keeps_override(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_trace_config(HIDE_ALL):
        span = tracer.start_span("stream", openinference_span_kind="llm")
        span.set_input("secret-in")
    # Simulates a streamed response finishing after the block exits.
    span.set_output("secret-out")
    span.end()
    attrs = _attrs(in_memory_span_exporter)
    assert attrs[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attrs[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE


def test_span_started_outside_block_ignores_override(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    span = tracer.start_span("pre", openinference_span_kind="chain")
    with using_trace_config(HIDE_ALL):
        span.set_input("secret-in")
    span.end()
    assert _attrs(in_memory_span_exporter)[SpanAttributes.INPUT_VALUE] == "secret-in"


def _llm_span_with_content(tracer: OITracer, kind: str) -> None:
    span = tracer.start_span("genai", openinference_span_kind=kind)  # type: ignore[arg-type]
    span.set_attributes(
        {
            SpanAttributes.INPUT_VALUE: json.dumps({"q": "SECRET_IN"}),
            SpanAttributes.INPUT_MIME_TYPE: "application/json",
            SpanAttributes.OUTPUT_VALUE: "SECRET_OUT",
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "SECRET_SYS",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "SECRET_IN",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "SECRET_OUT",
            "llm.tools.0.tool.json_schema": json.dumps(
                {"type": "function", "function": {"name": "f", "description": "SECRET_TOOL"}}
            ),
            SpanAttributes.TOOL_NAME: "f",
            SpanAttributes.TOOL_PARAMETERS: json.dumps({"arg": "SECRET_ARG"}),
            "retrieval.documents.0.document.content": "SECRET_DOC",
            SpanAttributes.LLM_MODEL_NAME: "gpt-4o",
            SpanAttributes.LLM_PROVIDER: "openai",
        }
    )
    span.end()


@pytest.mark.parametrize("kind", ["llm", "tool", "retriever"])
def test_genai_content_keys_masked_by_override(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter, kind: str
) -> None:
    with using_trace_config(
        TraceConfig(hide_inputs=True, hide_outputs=True, enable_genai_semconv=True)
    ):
        _llm_span_with_content(tracer, kind)
    attrs = _attrs(in_memory_span_exporter)
    genai = {k: v for k, v in attrs.items() if k.startswith("gen_ai.")}
    assert genai.get("gen_ai.request.model") == "gpt-4o"
    for key, value in genai.items():
        assert "SECRET" not in str(value), key
    for key in (
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
    ):
        assert key not in attrs


@pytest.mark.parametrize(
    "trace_config",
    [TraceConfig(hide_inputs=True, hide_outputs=True, enable_genai_semconv=True)],
)
def test_genai_content_keys_masked_by_instrumentor_config(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    span = tracer.start_span("direct", openinference_span_kind="llm")
    span.set_attribute("gen_ai.input.messages", "SECRET_DIRECT")
    span.set_attribute("gen_ai.output.messages", "SECRET_DIRECT")
    span.set_attribute("gen_ai.tool.call.arguments", "SECRET_DIRECT")
    span.end()
    attrs = _attrs(in_memory_span_exporter)
    assert "gen_ai.input.messages" not in attrs
    assert "gen_ai.output.messages" not in attrs
    assert attrs["gen_ai.tool.call.arguments"] == REDACTED_VALUE


def test_genai_content_emitted_without_hiding(
    tracer: OITracer, in_memory_span_exporter: InMemorySpanExporter
) -> None:
    with using_trace_config(TraceConfig(enable_genai_semconv=True)):
        _llm_span_with_content(tracer, "llm")
    attrs = _attrs(in_memory_span_exporter)
    assert "SECRET_IN" in str(attrs["gen_ai.input.messages"])
    assert "SECRET_OUT" in str(attrs["gen_ai.output.messages"])
