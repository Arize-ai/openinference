from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import pytest
from autogen import ConversableAgent
from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import TraceConfig, using_attributes
from openinference.instrumentation.ag2 import AG2Instrumentor
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)


def _agent(name: str, default_auto_reply: str = "done") -> ConversableAgent:
    return ConversableAgent(
        name,
        llm_config=False,
        code_execution_config=False,
        human_input_mode="NEVER",
        default_auto_reply=default_auto_reply,
    )


def _attributes(span: ReadableSpan) -> Mapping[str, AttributeValue]:
    return cast(Mapping[str, AttributeValue], span.attributes)


def test_entrypoints() -> None:
    for group in ("opentelemetry_instrumentor", "openinference_instrumentor"):
        (entrypoint,) = entry_points(group=group, name="ag2")
        assert isinstance(entrypoint.load()(), AG2Instrumentor)


def test_instrumentation_dependencies() -> None:
    assert tuple(AG2Instrumentor().instrumentation_dependencies()) == ("ag2 >= 0.14.0, < 1.0.0",)


def test_generate_reply_has_agent_and_context_attributes(
    instrumentor: AG2Instrumentor,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    agent = _agent("writer", "draft")
    messages = [{"role": "user", "content": "Write a title"}]

    with using_attributes(
        session_id="session-1",
        user_id="user-1",
        metadata={"tenant": "acme"},
        tags=["test"],
    ):
        assert agent.generate_reply(messages=messages) == "draft"

    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = _attributes(span)
    assert span.name == "writer.generate_reply"
    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == (
        OpenInferenceSpanKindValues.AGENT.value
    )
    assert attributes[SpanAttributes.AGENT_NAME] == "writer"
    assert attributes[SpanAttributes.SESSION_ID] == "session-1"
    assert attributes[SpanAttributes.USER_ID] == "user-1"
    assert json.loads(cast(str, attributes[SpanAttributes.INPUT_VALUE])) == messages
    assert json.loads(cast(str, attributes[SpanAttributes.OUTPUT_VALUE])) == "draft"
    assert span.status.status_code is StatusCode.OK


def test_unserializable_output_does_not_change_agent_behavior(
    instrumentor: AG2Instrumentor,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    class Unserializable:
        def model_dump(self, **kwargs: object) -> object:
            raise RuntimeError("cannot serialize")

    output = Unserializable()
    agent = _agent("custom")
    cast(Any, agent).register_reply(
        lambda sender: sender is None,
        lambda *args, **kwargs: (True, output),
        position=0,
    )

    result = cast(Any, agent.generate_reply(messages=[{"role": "user", "content": "hello"}]))
    assert result is output
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert json.loads(cast(str, _attributes(span)[SpanAttributes.OUTPUT_VALUE])) == (
        "<unserializable>"
    )


@pytest.mark.asyncio
async def test_async_generate_reply(
    instrumentor: AG2Instrumentor,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    agent = _agent("reviewer", "approved")

    assert (
        await agent.a_generate_reply(messages=[{"role": "user", "content": "Review this"}])
        == "approved"
    )

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "reviewer.a_generate_reply"
    assert _attributes(span)[SpanAttributes.OPENINFERENCE_SPAN_KIND] == (
        OpenInferenceSpanKindValues.AGENT.value
    )


def test_sync_chat_parents_reply_span(
    instrumentor: AG2Instrumentor,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    sender = _agent("sender")
    recipient = _agent("recipient")

    result = sender.initiate_chat(recipient, message="hello", max_turns=1, silent=True)

    spans = in_memory_span_exporter.get_finished_spans()
    chat_span = next(span for span in spans if span.name == "sender.initiate_chat")
    reply_span = next(span for span in spans if span.name == "recipient.generate_reply")
    assert result.chat_history[-1]["content"] == "done"
    assert reply_span.parent is not None
    assert reply_span.parent.span_id == chat_span.context.span_id
    assert _attributes(chat_span)[SpanAttributes.OPENINFERENCE_SPAN_KIND] == (
        OpenInferenceSpanKindValues.CHAIN.value
    )


@pytest.mark.asyncio
async def test_async_chat_parents_reply_span(
    instrumentor: AG2Instrumentor,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    sender = _agent("sender")
    recipient = _agent("recipient")

    result = await sender.a_initiate_chat(recipient, message="hello", max_turns=1, silent=True)

    spans = in_memory_span_exporter.get_finished_spans()
    chat_span = next(span for span in spans if span.name == "sender.a_initiate_chat")
    reply_span = next(span for span in spans if span.name == "recipient.a_generate_reply")
    assert result.chat_history[-1]["content"] == "done"
    assert reply_span.parent is not None
    assert reply_span.parent.span_id == chat_span.context.span_id


def test_tool_success_and_failure(
    instrumentor: AG2Instrumentor,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    agent = _agent("executor")

    def add(a: int, b: int) -> int:
        return a + b

    agent.register_function({"add": add})
    success, output = agent.execute_function(
        {"name": "add", "arguments": '{"a": 2, "b": 3}'}, call_id="call-1"
    )
    missing, _ = agent.execute_function({"name": "missing", "arguments": "{}"})

    assert success is True
    assert output["content"] == 5
    assert missing is False
    add_span, missing_span = in_memory_span_exporter.get_finished_spans()
    attributes = _attributes(add_span)
    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == (
        OpenInferenceSpanKindValues.TOOL.value
    )
    assert attributes[ToolCallAttributes.TOOL_CALL_ID] == "call-1"
    assert json.loads(
        cast(str, attributes[ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON])
    ) == {"a": 2, "b": 3}
    assert add_span.status.status_code is StatusCode.OK
    assert missing_span.status.status_code is StatusCode.ERROR


@pytest.mark.asyncio
async def test_async_tool_execution(
    instrumentor: AG2Instrumentor,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    agent = _agent("executor")

    async def multiply(a: int, b: int) -> int:
        return a * b

    agent.register_function({"multiply": multiply})
    success, output = await agent.a_execute_function(
        {"name": "multiply", "arguments": '{"a": 4, "b": 5}'}
    )

    assert success is True
    assert output["content"] == 20
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.name == "multiply"
    assert span.status.status_code is StatusCode.OK


def test_suppression(
    instrumentor: AG2Instrumentor,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with suppress_instrumentation():
        _agent("quiet").generate_reply(messages=[{"role": "user", "content": "hello"}])

    assert in_memory_span_exporter.get_finished_spans() == ()


def test_trace_config_masks_input_and_output(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    instrumentor = AG2Instrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )
    try:
        _agent("private", "secret output").generate_reply(
            messages=[{"role": "user", "content": "secret input"}]
        )
    finally:
        instrumentor.uninstrument()

    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = _attributes(span)
    assert attributes[SpanAttributes.INPUT_VALUE] == "__REDACTED__"
    assert attributes[SpanAttributes.OUTPUT_VALUE] == "__REDACTED__"


def test_uninstrument_restores_methods(tracer_provider: TracerProvider) -> None:
    original = ConversableAgent.generate_reply
    instrumentor = AG2Instrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    assert ConversableAgent.generate_reply is not original
    instrumentor.uninstrument()
    assert ConversableAgent.generate_reply is original
