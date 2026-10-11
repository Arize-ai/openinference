"""Tests that the root (trace-level) span gets ERROR status when the agent run raises."""

from __future__ import annotations

from typing import Any

import pytest
from agents import Agent, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor


@function_tool
def get_weather(city: str) -> str:
    """Get weather."""
    return f"sunny in {city}"


class _LoopForeverModel(Model):
    """Model that always asks for a tool call, causing MaxTurnsExceeded."""

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        output: list[Any] = [
            ResponseFunctionToolCall(
                type="function_call",
                call_id="call-1",
                name="get_weather",
                arguments='{"city": "London"}',
            )
        ]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _SuccessModel(Model):
    """Model that succeeds on the first turn."""

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        output: list[Any] = [
            ResponseOutputMessage(
                id="m1",
                type="message",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text="Hello!", annotations=[])],
            )
        ]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class _ToolRaisesModel(Model):
    """Model that calls a tool on turn 1 (tool raises), then answers on turn 2."""

    def __init__(self) -> None:
        self._calls = 0

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self._calls += 1
        if self._calls == 1:
            output: list[Any] = [
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call-1",
                    name="get_weather_raises",
                    arguments='{"city": "London"}',
                )
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id="m1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(type="output_text", text="It is sunny.", annotations=[])
                    ],
                )
            ]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def exporter() -> Any:
    _exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_exporter))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    yield _exporter
    OpenAIAgentsInstrumentor().uninstrument()


def _root_span(spans: list[ReadableSpan]) -> ReadableSpan:
    roots = [s for s in spans if s.parent is None]
    assert len(roots) == 1, f"expected 1 root span, got {len(roots)}: {[s.name for s in roots]}"
    return roots[0]


async def test_root_span_is_error_when_max_turns_exceeded(exporter: InMemorySpanExporter) -> None:
    """When MaxTurnsExceeded is raised the root span must be ERROR."""
    agent = Agent(name="helper", model=_LoopForeverModel(), tools=[get_weather])
    with pytest.raises(MaxTurnsExceeded):
        await Runner.run(agent, "hi", max_turns=2)

    spans = list(exporter.get_finished_spans())
    root = _root_span(spans)
    assert root.status.status_code == StatusCode.ERROR, (
        f"root span status should be ERROR, got {root.status.status_code}"
    )


async def test_root_span_is_ok_when_run_succeeds(exporter: InMemorySpanExporter) -> None:
    """A successful run must keep the root span as OK (regression guard)."""
    agent = Agent(name="helper", model=_SuccessModel())
    result = await Runner.run(agent, "hi")
    assert result.final_output == "Hello!"

    spans = list(exporter.get_finished_spans())
    root = _root_span(spans)
    assert root.status.status_code == StatusCode.OK, (
        f"root span status should be OK, got {root.status.status_code}"
    )


async def test_root_span_is_ok_when_tool_raises_but_run_succeeds(
    exporter: InMemorySpanExporter,
) -> None:
    """A non-fatal tool error must NOT mark the root span as ERROR."""

    @function_tool
    def get_weather_raises(city: str) -> str:
        """Get weather but always raises."""
        raise RuntimeError("Tool execution failed")

    agent = Agent(name="helper", model=_ToolRaisesModel(), tools=[get_weather_raises])
    result = await Runner.run(agent, "hi")
    assert result.final_output == "It is sunny."

    spans = list(exporter.get_finished_spans())
    root = _root_span(spans)
    assert root.status.status_code == StatusCode.OK, (
        f"root span status should be OK for non-fatal tool error, got {root.status.status_code}"
    )
