"""Tests for how the request is recorded: its parameters in input.value and the invocation
parameters, and a tool result in its input messages."""

import json
from typing import Any, Callable, Dict, Iterable, List

import httpx2
import pytest
from anthropic import NOT_GIVEN, Anthropic, omit
from anthropic.types import ToolUseBlock
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from openinference.instrumentation.anthropic._wrappers import (
    _get_inputs,
    _get_llm_input_messages,
)
from openinference.semconv.trace import (
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
INPUT_VALUE = SpanAttributes.INPUT_VALUE
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT

MODEL = "claude-sonnet-4-6"


def _mock_anthropic_client() -> Anthropic:
    def handler(request: Any) -> Any:
        return httpx2.Response(
            status_code=200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": MODEL,
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        )

    transport = httpx2.MockTransport(handler)
    return Anthropic(api_key="sk-ant-fake", http_client=httpx2.Client(transport=transport))


def _span_attributes(exporter: InMemorySpanExporter) -> Dict[str, Any]:
    (span,) = exporter.get_finished_spans()
    return dict(span.attributes or {})


def test_params_not_given_are_not_recorded_and_models_are_dumped(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    # the tool runner passes the parameters that were not given as `omit` explicitly, and
    # passes the response's blocks back as models
    tool_use = ToolUseBlock(id="toolu_1", name="f", input={"a": 1}, type="tool_use")
    messages: List[Any] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [tool_use]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1"}]},
    ]
    _mock_anthropic_client().messages.create(
        model=MODEL,
        max_tokens=10,
        stop_sequences=omit,
        metadata=omit,
        extra_query=None,
        extra_body=None,
        timeout=NOT_GIVEN,
        messages=messages,
    )
    attributes = _span_attributes(in_memory_span_exporter)
    assert json.loads(attributes[LLM_INVOCATION_PARAMETERS]) == {"max_tokens": 10}
    input_value = json.loads(attributes[INPUT_VALUE])
    assert set(input_value) == {"model", "max_tokens", "messages"}
    assert input_value["messages"][1]["content"] == [
        {"id": "toolu_1", "name": "f", "input": {"a": 1}, "type": "tool_use"}
    ]


def test_input_value_records_a_model_class_output_format() -> None:
    class City(BaseModel):
        name: str

    attributes = dict(
        _get_inputs(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                # a model class, which has no instance to dump
                "output_format": City,
            }
        )
    )
    assert "City" in json.loads(attributes[INPUT_VALUE])["output_format"]


def _tool_result(tool_use_id: str, content: Any) -> Dict[str, Any]:
    return {"type": "tool_result", "tool_use_id": tool_use_id, "content": content}


def test_one_tool_result_is_recorded_as_the_message_content() -> None:
    attributes = dict(
        _get_llm_input_messages(
            [{"role": "user", "content": [_tool_result("toolu_1", "sunny")]}],
        )
    )
    prefix = f"{LLM_INPUT_MESSAGES}.0"
    assert attributes == {
        f"{prefix}.{MESSAGE_ROLE}": "tool",
        f"{prefix}.{MESSAGE_TOOL_CALL_ID}": "toolu_1",
        f"{prefix}.{MESSAGE_CONTENT}": "sunny",
    }


@pytest.mark.parametrize("container", [list, tuple, iter])
@pytest.mark.parametrize("with_tool_results", [False, True])
def test_input_content_iterables_are_preserved(
    container: Callable[[Iterable[Any]], Iterable[Any]], with_tool_results: bool
) -> None:
    blocks: List[Any] = [{"type": "text", "text": "Now answer."}]
    if with_tool_results:
        blocks = [_tool_result("toolu_1", "4"), _tool_result("toolu_2", "sunny")] + blocks
    messages: List[Any] = [
        {"role": "user", "content": container(blocks)},
        {"role": "assistant", "content": "done"},
    ]
    attributes = dict(_get_llm_input_messages(messages, system="be brief"))
    prefix = LLM_INPUT_MESSAGES
    index = 3 if with_tool_results else 1
    expected = {
        f"{prefix}.0.{MESSAGE_ROLE}": "system",
        f"{prefix}.0.{MESSAGE_CONTENT}": "be brief",
        f"{prefix}.{index}.{MESSAGE_ROLE}": "user",
        f"{prefix}.{index}.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}": "text",
        f"{prefix}.{index}.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}": "Now answer.",
        f"{prefix}.{index + 1}.{MESSAGE_ROLE}": "assistant",
        f"{prefix}.{index + 1}.{MESSAGE_CONTENT}": "done",
    }
    if with_tool_results:
        for i, (tool_id, result) in enumerate([("toolu_1", "4"), ("toolu_2", "sunny")], 1):
            expected.update(
                {
                    f"{prefix}.{i}.{MESSAGE_ROLE}": "tool",
                    f"{prefix}.{i}.{MESSAGE_TOOL_CALL_ID}": tool_id,
                    f"{prefix}.{i}.{MESSAGE_CONTENT}": result,
                }
            )
    assert attributes == expected
