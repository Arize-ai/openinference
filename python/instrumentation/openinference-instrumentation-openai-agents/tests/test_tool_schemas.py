"""Tests for tool.description / tool.parameters on function spans.

The SDK's FunctionSpanData carries only a tool's name, input and output, so the schema is
read off the live FunctionTool being invoked. Because that depends on patching a private
SDK step whose location and signature have changed between releases, the important test
here is the end-to-end one: it runs a real Agent through a fake model and asserts the
attributes land on the exported span.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from agents import Agent, Runner, function_tool, tool_namespace
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

from openinference.instrumentation import (
    suppress_tracing,
    using_metadata,
    using_session,
    using_tags,
    using_user,
)
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


def _plain_tool_call() -> Any:
    return ResponseFunctionToolCall(
        type="function_call",
        call_id="call-1",
        name="get_weather",
        arguments='{"city":"London"}',
    )


class _FakeModel(Model):
    """Calls get_weather on the first turn, then answers.

    ``tool_call`` builds the call the model asks for, so a test can vary its wire shape.
    """

    def __init__(self, tool_call: Callable[[], Any] = _plain_tool_call) -> None:
        self.calls = 0
        self.tool_call = tool_call

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            output: list[Any] = [self.tool_call()]
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


async def test_namespaced_tool_reports_tool_schema_end_to_end(
    exporter_and_instrumentation: InMemorySpanExporter,
) -> None:
    """The SDK names the span "<namespace>.<name>" while FunctionTool.name stays bare, so
    keying the schemas by the tool name alone loses both attributes here."""
    exporter = exporter_and_instrumentation
    # Widened because Agent.tools is an invariant list of the full Tool union.
    tools: list[Any] = list(
        tool_namespace(name="weather", description="Weather tools", tools=[get_weather])
    )

    def namespaced_call() -> Any:
        # The namespace travels as its own field on the tool call, not inside the name.
        call = ResponseFunctionToolCall.model_construct(
            type="function_call",
            call_id="call-1",
            name="get_weather",
            arguments='{"city":"London"}',
        )
        object.__setattr__(call, "namespace", "weather")
        return call

    agent = Agent(name="WeatherAgent", model=_FakeModel(namespaced_call), tools=tools)

    result = await Runner.run(agent, "What's the weather in London?")
    assert result.final_output == "It is 21C and sunny."

    attrs = _tool_span(list(exporter.get_finished_spans()))
    assert attrs["tool.name"] == "weather.get_weather"
    assert attrs["tool.description"] == _DESCRIPTION
    parameters = json.loads(str(attrs["tool.parameters"]))
    assert "city" in parameters["properties"]


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


# --- keying: the span name, not the tool name ---------------------------------------


def test_namespaced_tool_is_keyed_by_the_name_its_span_will_carry() -> None:
    """From 0.11 the SDK names a namespaced tool's span "<namespace>.<name>"."""
    result = schemas_from_tool_runs(
        [_ToolRun(_Tool("get", "d", params_json_schema={}, _tool_namespace="weather"))]
    )
    assert result["weather.get"] == ("d", "{}")


def test_namespaced_tool_does_not_claim_the_bare_key() -> None:
    """A plain tool of the same name must not be handed the namespaced tool's schema.

    The SDK allows both on one agent, and the plain tool's span is named bare.
    """
    result = schemas_from_tool_runs(
        [
            _ToolRun(_Tool("get", "plain", params_json_schema={})),
            _ToolRun(_Tool("get", "namespaced", params_json_schema={}, _tool_namespace="weather")),
        ]
    )
    assert result["get"][0] == "plain"
    assert result["weather.get"][0] == "namespaced"


def test_same_tool_name_in_two_namespaces_keeps_both_schemas() -> None:
    """Bare-name keying collapsed these onto one entry; the SDK permits both."""
    result = schemas_from_tool_runs(
        [
            _ToolRun(_Tool("get", "w", params_json_schema={}, _tool_namespace="weather")),
            _ToolRun(_Tool("get", "c", params_json_schema={}, _tool_namespace="calendar")),
        ]
    )
    assert result["weather.get"][0] == "w"
    assert result["calendar.get"][0] == "c"


def test_reserved_synthetic_namespace_is_keyed_by_the_bare_name() -> None:
    """A deferred top-level tool sets namespace == name, and the SDK keeps the span bare."""
    result = schemas_from_tool_runs(
        [_ToolRun(_Tool("get", "d", params_json_schema={}, _tool_namespace="get"))]
    )
    assert result["get"] == ("d", "{}")


def test_namespace_of_an_unexpected_type_falls_back_to_the_bare_name() -> None:
    result = schemas_from_tool_runs(
        [_ToolRun(_Tool("get", "d", params_json_schema={}, _tool_namespace=object()))]
    )
    assert result["get"] == ("d", "{}")


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


# --- suppress tracing ---------------------------------------------------------------


async def test_no_spans_when_tracing_suppressed(
    exporter_and_instrumentation: InMemorySpanExporter,
) -> None:
    """Inside suppress_tracing() no span is exported, and the run still works.

    The patched execution step is on the SDK's real tool-calling path, so this also
    asserts the suppression guard does not interfere with the tool call itself.
    """
    exporter = exporter_and_instrumentation
    agent = Agent(name="WeatherAgent", model=_FakeModel(), tools=[get_weather])

    with suppress_tracing():
        result = await Runner.run(agent, "What's the weather in London?")

    assert result.final_output == "It is 21C and sunny."
    assert exporter.get_finished_spans() == ()


async def test_wrapper_publishes_nothing_when_tracing_suppressed() -> None:
    """The wrapper must skip its work outright, not rely on the span being dropped.

    Publishing schemas under suppression is wasted serialization for attributes that a
    non-recording span discards, so the ContextVar must stay untouched.
    """
    wrapper = make_execute_function_tools_wrapper()
    seen: dict[str, Any] = {}

    async def wrapped(**kwargs: Any) -> str:
        seen["inside"] = get_tool_schema("t")
        return "done"

    tool_runs = [_ToolRun(_Tool("t", "d", params_json_schema={}))]
    with suppress_tracing():
        assert await wrapper(wrapped, None, (), {"tool_runs": tool_runs}) == "done"
    assert seen["inside"] is None

    # Outside suppression the same call publishes, proving the guard is what skipped it.
    await wrapper(wrapped, None, (), {"tool_runs": tool_runs})
    assert seen["inside"] == ("d", "{}")


# --- context attribute propagation --------------------------------------------------


async def test_context_attributes_propagate_to_function_spans(
    exporter_and_instrumentation: InMemorySpanExporter,
) -> None:
    """Context attributes must land on function spans alongside the tool schema."""
    exporter = exporter_and_instrumentation
    agent = Agent(name="WeatherAgent", model=_FakeModel(), tools=[get_weather])

    with (
        using_session("s-1"),
        using_user("u-1"),
        using_metadata({"k": "v"}),
        using_tags(["t1", "t2"]),
    ):
        await Runner.run(agent, "What's the weather in London?")

    attrs = _tool_span(list(exporter.get_finished_spans()))
    assert attrs["session.id"] == "s-1"
    assert attrs["user.id"] == "u-1"
    assert json.loads(str(attrs["metadata"])) == {"k": "v"}
    assert list(attrs["tag.tags"]) == ["t1", "t2"]
    # The schema enrichment and the context attributes must coexist on the same span.
    assert attrs["tool.description"] == _DESCRIPTION
    assert attrs["tool.parameters"]
