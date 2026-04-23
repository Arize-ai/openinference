"""
Integration tests for GenAI emission on the OITracer.

When ``TraceConfig(enable_genai_semconv=True)`` or the
``OPENINFERENCE_ENABLE_GENAI_SEMCONV`` environment variable is set, the
OITracer writes gen_ai.* attributes onto the span alongside the existing
OpenInference attributes.
"""

import json
from typing import cast

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import OITracer, TraceConfig, TracerProvider
from openinference.instrumentation.config import OPENINFERENCE_ENABLE_GENAI_SEMCONV
from openinference.instrumentation.genai.attributes import GenAIAttributes as GA
from openinference.semconv.trace import (
    MessageAttributes,
    SpanAttributes,
)


def _make_tracer(
    exporter: InMemorySpanExporter,
    *,
    enable: bool,
) -> OITracer:
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    config = TraceConfig(enable_genai_semconv=enable)
    provider = TracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer(__name__)


def test_genai_disabled_by_default(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = _make_tracer(in_memory_span_exporter, enable=False)
    tracer.start_span(
        "chat",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
            SpanAttributes.LLM_MODEL_NAME: "gpt-4o",
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 5,
        },
    ).end()
    span = in_memory_span_exporter.get_finished_spans()[0]
    assert span.attributes is not None
    assert GA.GEN_AI_REQUEST_MODEL not in span.attributes
    assert GA.GEN_AI_USAGE_INPUT_TOKENS not in span.attributes
    # OI attributes remain
    assert span.attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"


def test_genai_enabled_emits_both(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = _make_tracer(in_memory_span_exporter, enable=True)
    base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    tracer.start_span(
        "chat",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
            SpanAttributes.LLM_MODEL_NAME: "gpt-4o",
            SpanAttributes.LLM_SYSTEM: "openai",
            SpanAttributes.LLM_PROVIDER: "openai",
            SpanAttributes.LLM_TOKEN_COUNT_PROMPT: 7,
            SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: 13,
            SpanAttributes.LLM_INVOCATION_PARAMETERS: json.dumps({"temperature": 0.5}),
            f"{base}.{MessageAttributes.MESSAGE_ROLE}": "user",
            f"{base}.{MessageAttributes.MESSAGE_CONTENT}": "Hi",
        },
    ).end()
    span = in_memory_span_exporter.get_finished_spans()[0]
    attrs = span.attributes or {}
    # OI attributes are preserved
    assert attrs.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"
    assert attrs.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 7
    # GenAI equivalents are emitted
    assert attrs.get(GA.GEN_AI_OPERATION_NAME) == "chat"
    assert attrs.get(GA.GEN_AI_REQUEST_MODEL) == "gpt-4o"
    assert attrs.get(GA.GEN_AI_PROVIDER_NAME) == "openai"
    assert attrs.get(GA.GEN_AI_USAGE_INPUT_TOKENS) == 7
    assert attrs.get(GA.GEN_AI_USAGE_OUTPUT_TOKENS) == 13
    assert attrs.get(GA.GEN_AI_REQUEST_TEMPERATURE) == 0.5
    # Structured JSON messages
    messages = json.loads(cast(str, attrs[GA.GEN_AI_INPUT_MESSAGES]))
    assert messages == [{"role": "user", "parts": [{"type": "text", "content": "Hi"}]}]


def test_env_var_enables_emission(
    in_memory_span_exporter: InMemorySpanExporter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting the env var and constructing a default TraceConfig enables emission."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    monkeypatch.setenv(OPENINFERENCE_ENABLE_GENAI_SEMCONV, "true")
    config = TraceConfig()
    assert config.enable_genai_semconv is True

    provider = TracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    tracer = provider.get_tracer(__name__)
    tracer.start_span(
        "chat",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
            SpanAttributes.LLM_MODEL_NAME: "claude-4-7",
        },
    ).end()
    attrs = in_memory_span_exporter.get_finished_spans()[0].attributes or {}
    assert attrs.get(GA.GEN_AI_REQUEST_MODEL) == "claude-4-7"
    assert attrs.get(GA.GEN_AI_OPERATION_NAME) == "chat"


def test_genai_respects_masking(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """When hide_input_messages is set, GenAI messages must not appear either."""
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    config = TraceConfig(enable_genai_semconv=True, hide_input_messages=True)
    provider = TracerProvider(config=config)
    provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    tracer = provider.get_tracer(__name__)
    base = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0"
    tracer.start_span(
        "chat",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "LLM",
            SpanAttributes.LLM_MODEL_NAME: "gpt-4o",
            f"{base}.{MessageAttributes.MESSAGE_ROLE}": "user",
            f"{base}.{MessageAttributes.MESSAGE_CONTENT}": "secret",
        },
    ).end()
    attrs = in_memory_span_exporter.get_finished_spans()[0].attributes or {}
    # hide_input_messages drops the OI flattened messages, which means the
    # genai mapper doesn't see them and emits nothing either.
    assert GA.GEN_AI_INPUT_MESSAGES not in attrs
    # Non-sensitive GenAI attributes still emit.
    assert attrs.get(GA.GEN_AI_REQUEST_MODEL) == "gpt-4o"


def test_tool_execution_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    tracer = _make_tracer(in_memory_span_exporter, enable=True)
    tracer.start_span(
        "execute_tool",
        attributes={
            SpanAttributes.OPENINFERENCE_SPAN_KIND: "TOOL",
            SpanAttributes.TOOL_NAME: "get_weather",
            SpanAttributes.TOOL_ID: "call_1",
            SpanAttributes.INPUT_VALUE: '{"city": "Paris"}',
            SpanAttributes.OUTPUT_VALUE: "Rainy, 57F",
        },
    ).end()
    attrs = in_memory_span_exporter.get_finished_spans()[0].attributes or {}
    assert attrs.get(GA.GEN_AI_OPERATION_NAME) == "execute_tool"
    assert attrs.get(GA.GEN_AI_TOOL_NAME) == "get_weather"
    assert attrs.get(GA.GEN_AI_TOOL_CALL_ID) == "call_1"
    assert attrs.get(GA.GEN_AI_TOOL_CALL_ARGUMENTS) == '{"city": "Paris"}'
    assert attrs.get(GA.GEN_AI_TOOL_CALL_RESULT) == "Rainy, 57F"


def test_default_config_has_flag_off() -> None:
    config = TraceConfig()
    assert config.enable_genai_semconv is False


def test_trace_config_flag_can_be_set() -> None:
    config = TraceConfig(enable_genai_semconv=True)
    assert config.enable_genai_semconv is True
