"""Tests for tool.description / tool.parameters on function spans.

The SDK's FunctionSpanData carries only a tool's name, input and output, so the schema is
read off the live FunctionTool being invoked. Because that depends on patching a private
SDK step whose location and signature have changed between releases, the important test
here is the end-to-end one: it runs a real Agent through a fake model and asserts the
attributes land on the exported span.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

try:
    from agents import Agent, Runner, function_tool
    from agents.items import ModelResponse
    from agents.models.interface import Model
    from agents.usage import Usage
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseOutputText,
    )
except ImportError:
    # Handle compatibility issue with OpenAI SDK >=1.103.0 where WebSearchToolFilters was removed
    # Introduced in: https://github.com/openai/openai-python/commit/3d3d16a
    pytest.skip(
        "agents package incompatible with current OpenAI SDK version", allow_module_level=True
    )

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openai_agents import (
    OpenAIAgentsInstrumentor,
    _patch_tool_execution,
)
from openinference.instrumentation.openai_agents._tool_schemas import (
    find_tool_execution_bindings,
    get_tool_schema,
    make_execute_function_tools_wrapper,
    schemas_from_tool_runs,
)

_DESCRIPTION = "Get the current weather for a city."


@function_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"21C and sunny in {city}"


class _FakeModel(Model):
    """Calls get_weather on the first turn, then answers."""

    def __init__(self) -> None:
        self.calls = 0

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            output: list[Any] = [
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call-1",
                    name="get_weather",
                    arguments='{"city":"London"}',
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
                        ResponseOutputText(
                            type="output_text", text="It is 21C and sunny.", annotations=[]
                        )
                    ],
                )
            ]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def _tool_span(spans: list[ReadableSpan]) -> dict[str, Any]:
    matching = [
        dict(s.attributes or {})
        for s in spans
        if (s.attributes or {}).get("openinference.span.kind") == "TOOL"
    ]
    assert len(matching) == 1, f"expected one TOOL span, got {len(matching)}"
    return matching[0]


# --- end to end through the real SDK ------------------------------------------------


@pytest.fixture
def exporter_and_instrumentation() -> Any:
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    yield exporter
    OpenAIAgentsInstrumentor().uninstrument()


async def test_function_span_reports_tool_schema_end_to_end(
    exporter_and_instrumentation: InMemorySpanExporter,
) -> None:
    """The schema must reach the exported span through the real SDK code path."""
    exporter = exporter_and_instrumentation
    agent = Agent(name="WeatherAgent", model=_FakeModel(), tools=[get_weather])

    result = await Runner.run(agent, "What's the weather in London?")
    assert result.final_output == "It is 21C and sunny."

    attrs = _tool_span(list(exporter.get_finished_spans()))
    assert attrs["tool.name"] == "get_weather"
    assert attrs["tool.description"] == _DESCRIPTION
    # The SDK derives the schema from the function signature.
    parameters = json.loads(str(attrs["tool.parameters"]))
    assert "city" in parameters["properties"]
    assert parameters["required"] == ["city"]


async def test_agent_name_recorded_end_to_end(
    exporter_and_instrumentation: InMemorySpanExporter,
) -> None:
    exporter = exporter_and_instrumentation
    agent = Agent(name="WeatherAgent", model=_FakeModel(), tools=[get_weather])
    await Runner.run(agent, "What's the weather in London?")

    agent_spans = [
        dict(s.attributes or {})
        for s in exporter.get_finished_spans()
        if (s.attributes or {}).get("agent.name")
    ]
    assert [a["agent.name"] for a in agent_spans] == ["WeatherAgent"]


# --- schema extraction --------------------------------------------------------------


class _Tool:
    def __init__(self, name: Any, description: Any = None, **extra: Any) -> None:
        self.name = name
        if description is not None:
            self.description = description
        for key, value in extra.items():
            setattr(self, key, value)


class _ToolRun:
    def __init__(self, function_tool: Any) -> None:
        self.function_tool = function_tool


def test_schemas_read_sdk_params_json_schema_field() -> None:
    """The SDK's FunctionTool names this params_json_schema, not parameters."""
    schema = {"type": "object", "properties": {"city": {"type": "string"}}}
    result = schemas_from_tool_runs(
        [_ToolRun(_Tool("get_weather", _DESCRIPTION, params_json_schema=schema))]
    )
    assert result["get_weather"][0] == _DESCRIPTION
    assert json.loads(str(result["get_weather"][1])) == schema


def test_schemas_accept_openai_parameters_field() -> None:
    """The OpenAI Responses FunctionTool of the same name calls it parameters."""
    schema = {"type": "object", "properties": {}}
    result = schemas_from_tool_runs([_ToolRun(_Tool("t", "d", parameters=schema))])
    assert json.loads(str(result["t"][1])) == schema


@pytest.mark.parametrize(
    "tool_runs",
    [
        pytest.param(None, id="none"),
        pytest.param([], id="empty"),
        pytest.param(object(), id="not-iterable"),
        pytest.param([object()], id="no-function-tool"),
        pytest.param([_ToolRun(_Tool(name=None))], id="name-not-a-string"),
    ],
)
def test_schemas_tolerate_unexpected_shapes(tool_runs: Any) -> None:
    """This reads a private SDK dataclass, so a surprise must not raise mid-tool-call."""
    assert schemas_from_tool_runs(tool_runs) == {}


def test_schema_without_description_records_only_parameters() -> None:
    result = schemas_from_tool_runs([_ToolRun(_Tool("t", params_json_schema={"type": "object"}))])
    assert result["t"][0] is None
    assert result["t"][1] is not None


# --- publication scope --------------------------------------------------------------


async def test_schemas_are_not_visible_outside_the_execution_step() -> None:
    """Nothing to evict or clean up: the value cannot outlive the step that set it."""
    wrapper = make_execute_function_tools_wrapper()
    seen: dict[str, Any] = {}

    async def wrapped(**kwargs: Any) -> str:
        seen["inside"] = get_tool_schema("t")
        return "done"

    assert get_tool_schema("t") is None
    result = await wrapper(
        wrapped, None, (), {"tool_runs": [_ToolRun(_Tool("t", "d", params_json_schema={}))]}
    )
    assert result == "done"
    assert seen["inside"] == ("d", "{}")
    assert get_tool_schema("t") is None


async def test_nested_execution_steps_do_not_hide_outer_tools() -> None:
    """An agent exposed as a tool runs a nested step; the outer tools stay visible."""
    wrapper = make_execute_function_tools_wrapper()
    seen: dict[str, Any] = {}

    async def inner(**kwargs: Any) -> None:
        seen["outer_from_inner"] = get_tool_schema("outer")
        seen["inner_from_inner"] = get_tool_schema("inner")

    async def outer(**kwargs: Any) -> None:
        await wrapper(
            inner, None, (), {"tool_runs": [_ToolRun(_Tool("inner", "i", params_json_schema={}))]}
        )
        seen["inner_after"] = get_tool_schema("inner")

    await wrapper(
        outer, None, (), {"tool_runs": [_ToolRun(_Tool("outer", "o", params_json_schema={}))]}
    )
    assert seen["outer_from_inner"] == ("o", "{}")
    assert seen["inner_from_inner"] == ("i", "{}")
    # The nested step's tools are gone once it returns.
    assert seen["inner_after"] is None


async def test_wrapper_passes_through_when_there_is_nothing_to_publish() -> None:
    wrapper = make_execute_function_tools_wrapper()

    async def wrapped(**kwargs: Any) -> str:
        return "passed"

    assert await wrapper(wrapped, None, (), {"tool_runs": []}) == "passed"


# --- patching -----------------------------------------------------------------------


def test_patch_targets_a_real_sdk_step_and_is_reversible() -> None:
    """Guards against the SDK moving this step without the binding scan finding it."""
    patched = _patch_tool_execution()
    assert patched, "no known tool execution step could be patched"
    try:
        for owner, attribute, original in patched:
            assert owner.__dict__[attribute] is not original
    finally:
        for owner, attribute, original in patched:
            setattr(owner, attribute, original)
    for owner, attribute, original in patched:
        assert owner.__dict__[attribute] is original


def test_every_binding_of_the_step_is_found() -> None:
    """The step is imported by name at its call sites, so several modules bind it.

    Patching only the defining module leaves the real caller untouched, which is why the
    end-to-end test above is the one that matters.
    """
    bindings = find_tool_execution_bindings()
    assert bindings, "no binding of the tool execution step was found"
    for owner, attribute in bindings:
        assert attribute in owner.__dict__


def test_uninstrument_restores_every_patched_binding() -> None:
    def snapshot() -> dict[str, Any]:
        return {
            f"{owner!r}.{attribute}": owner.__dict__[attribute]
            for owner, attribute in find_tool_execution_bindings()
        }

    before = snapshot()
    assert before, "no known tool execution step is present in this SDK version"
    instrumentor = OpenAIAgentsInstrumentor()
    instrumentor.instrument()
    during = snapshot()
    assert all(during[key] is not before[key] for key in before)
    instrumentor.uninstrument()
    assert snapshot() == before
