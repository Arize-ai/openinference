"""Tests for LangGraph tool call instrumentation to ensure tool calls appear in output traces."""

import json
from datetime import datetime, timezone
from uuid import uuid4

from openinference.instrumentation.langchain._tracer import (
    LLM_OUTPUT_MESSAGES,
    MESSAGE_CONTENT,
    MESSAGE_ROLE,
    MESSAGE_TOOL_CALLS,
    TOOL_CALL_FUNCTION_ARGUMENTS_JSON,
    TOOL_CALL_FUNCTION_NAME,
    TOOL_CALL_ID,
    _flatten,
    _parse_message_data,
    _tool_calls_from_llm_outputs,
)


def test_parse_message_data_with_langgraph_tool_calls() -> None:
    """Test _parse_message_data extracts tool calls from LangGraph-style message data."""
    message_data = {
        "id": ["langchain", "schema", "messages", "AIMessage"],
        "kwargs": {
            "content": "I'll help you with both requests.",
            "additional_kwargs": {},
            "tool_calls": [
                {"name": "get_weather", "args": {"city": "San Francisco"}, "id": "call_1"},
                {"name": "calculate_sum", "args": {"a": 5, "b": 3}, "id": "call_2"},
            ],
        },
    }

    # Parse the message data
    parsed_attributes = dict(_parse_message_data(message_data))

    # Verify role is extracted
    assert parsed_attributes[MESSAGE_ROLE] == "assistant"

    # Verify content is extracted
    assert parsed_attributes[MESSAGE_CONTENT] == "I'll help you with both requests."

    # Verify tool calls are extracted
    assert MESSAGE_TOOL_CALLS in parsed_attributes
    tool_calls = parsed_attributes[MESSAGE_TOOL_CALLS]

    assert len(tool_calls) == 2

    # Verify first tool call
    weather_call = next(tc for tc in tool_calls if tc.get(TOOL_CALL_FUNCTION_NAME) == "get_weather")
    assert weather_call[TOOL_CALL_ID] == "call_1"
    assert json.loads(weather_call[TOOL_CALL_FUNCTION_ARGUMENTS_JSON]) == {"city": "San Francisco"}

    # Verify second tool call
    sum_call = next(tc for tc in tool_calls if tc.get(TOOL_CALL_FUNCTION_NAME) == "calculate_sum")
    assert sum_call[TOOL_CALL_ID] == "call_2"
    assert json.loads(sum_call[TOOL_CALL_FUNCTION_ARGUMENTS_JSON]) == {"a": 5, "b": 3}


def test_tool_calls_from_llm_outputs() -> None:
    """Test _tool_calls_from_llm_outputs extracts tool calls from LLM run.outputs."""
    # Mimics run.outputs for an LLM run (generations -> first set -> message with tool_calls)
    outputs = {
        "generations": [
            [
                {
                    "message": {
                        "id": ["langchain", "schema", "messages", "AIMessage"],
                        "kwargs": {
                            "content": "I'll use the calculator.",
                            "tool_calls": [
                                {"name": "add", "args": {"a": 1, "b": 2}, "id": "call_abc"},
                            ],
                        },
                    }
                }
            ]
        ]
    }
    attrs = dict(_flatten(_tool_calls_from_llm_outputs(outputs)))
    # Should produce flattened keys like llm.output_messages.0.message.tool_calls.0.*
    prefix = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0"
    assert f"{prefix}.{TOOL_CALL_ID}" in attrs
    assert attrs[f"{prefix}.{TOOL_CALL_ID}"] == "call_abc"
    assert f"{prefix}.{TOOL_CALL_FUNCTION_NAME}" in attrs
    assert attrs[f"{prefix}.{TOOL_CALL_FUNCTION_NAME}"] == "add"
    assert f"{prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}" in attrs
    assert json.loads(attrs[f"{prefix}.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"]) == {
        "a": 1,
        "b": 2,
    }


def test_tool_calls_propagated_to_parent_span() -> None:
    """Test that ending an LLM run with tool_calls sets those attributes on the parent span."""
    from langchain_core.tracers.schemas import Run
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk import trace as trace_sdk

    from openinference.instrumentation.langchain._tracer import OpenInferenceTracer

    # Capture attributes per span name so we assert tool calls are on parent, not LLM span.
    attrs_by_span_name = {}

    class CapturingSpanProcessor(trace_sdk.SpanProcessor):
        def on_end(self, span: trace_sdk.ReadableSpan) -> None:
            if hasattr(span, "name") and hasattr(span, "attributes") and span.attributes:
                name = getattr(span, "name", None) or str(id(span))
                attrs_by_span_name[name] = dict(span.attributes)

    tracer_provider = trace_sdk.TracerProvider()
    tracer_provider.add_span_processor(CapturingSpanProcessor())
    otel_tracer = trace_api.get_tracer("test", "1.0", tracer_provider)
    tracer = OpenInferenceTracer(otel_tracer, separate_trace_from_runtime_context=False)

    parent_run_id = uuid4()
    llm_run_id = uuid4()
    now = datetime.now(timezone.utc)
    parent_run = Run(
        id=parent_run_id,
        name="Agent",
        run_type="chain",
        inputs={},
        outputs=None,
        parent_run_id=None,
        start_time=now,
        end_time=now,
        error=None,
        serialized={},
        extra={},
        events=[],
    )
    llm_outputs = {
        "generations": [
            [
                {
                    "message": {
                        "id": ["langchain", "schema", "messages", "AIMessage"],
                        "kwargs": {
                            "content": "Calling get_weather.",
                            "tool_calls": [
                                {
                                    "name": "get_weather",
                                    "args": {"city": "Boston"},
                                    "id": "call_xyz",
                                },
                            ],
                        },
                    }
                }
            ]
        ]
    }
    llm_run = Run(
        id=llm_run_id,
        name="ChatOpenAI",
        run_type="llm",
        inputs={},
        outputs=llm_outputs,
        parent_run_id=parent_run_id,
        start_time=now,
        end_time=now,
        error=None,
        serialized={},
        extra={},
        events=[],
    )

    tracer._start_trace(parent_run)
    tracer._start_trace(llm_run)
    tracer._end_trace(llm_run)  # This propagates tool calls to parent span
    tracer._end_trace(parent_run)  # End parent so processor sees it with attributes

    prefix = f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0"
    parent_attrs = attrs_by_span_name.get("Agent", {})
    assert any(prefix in k for k in parent_attrs), (
        f"Expected parent span 'Agent' to have attributes containing {prefix!r}; "
        f"Agent keys: {list(parent_attrs.keys())}"
    )
