import json
import uuid
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from crewai import Task
from crewai.flow.flow import Flow, start  # type: ignore[import-untyped, unused-ignore]
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from pydantic import BaseModel, ConfigDict, Field

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.crewai._wrappers import (
    _get_execute_core_span_name,
    _get_input_value,
)
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

from ._scenarios import kickoff_agent, kickoff_crew, kickoff_flow, kickoff_flow_with_crew
from ._span_helpers import (
    GRAPH_NODE_ID,
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    JSON,
    LLM_PROMPT_TEMPLATE,
    LLM_PROMPT_TEMPLATE_VARIABLES,
    LLM_PROMPT_TEMPLATE_VERSION,
    METADATA,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    SESSION_ID,
    TAG_TAGS,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    TOOL_PARAMETERS,
    USER_ID,
    get_spans_by_kind,
)


def _pop_input_payload(attributes: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(str(attributes.pop(INPUT_VALUE)))
    assert isinstance(payload, dict)
    return cast(dict[str, Any], payload)


def _assert_serialized_agent_payload(
    payload: dict[str, Any],
    *,
    role: str,
    goal: str,
    backstory: str,
    allow_delegation: bool,
    verbose: bool,
    max_iter: int,
    tool_names: list[str],
) -> None:
    assert isinstance(payload["id"], str) and uuid.UUID(payload["id"])
    assert isinstance(payload["key"], str) and payload["key"]
    assert payload["role"] == role
    assert payload["goal"] == goal
    assert payload["backstory"] == backstory
    assert payload["verbose"] == verbose
    assert payload["allow_delegation"] == allow_delegation
    assert payload["max_iter"] == max_iter
    assert payload["max_rpm"] is None
    tools_payload = payload["tools"]
    assert isinstance(tools_payload, list)
    assert [tool["name"] for tool in tools_payload] == tool_names
    for tool in tools_payload:
        assert isinstance(tool, dict)
        assert "args_schema" not in tool
        assert "cache_function" not in tool
    assert "crew" not in payload
    assert "llm" not in payload
    assert "agent_executor" not in payload
    assert "executor_class" not in payload
    assert "tools_handler" not in payload
    assert "callbacks" not in payload
    assert "step_callback" not in payload
    assert "guardrail" not in payload
    assert "function_calling_llm" not in payload


def test_entrypoint_for_opentelemetry_instrument() -> None:
    """Test that the instrumentor is properly registered and implements OITracer."""
    instrumentor_entrypoints = list(
        entry_points(
            group="opentelemetry_instrumentor",
            name="crewai",
        )
    )
    assert len(instrumentor_entrypoints) == 1
    instrumentor = instrumentor_entrypoints[0].load()()
    assert isinstance(instrumentor, CrewAIInstrumentor)
    assert isinstance(CrewAIInstrumentor()._tracer, OITracer)


def test_get_input_value_serializes_agent_argument_without_cyclic_crew() -> None:
    class _AgentLike(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        role: str = "Tech Content Strategist"
        goal: str = "Craft compelling content on tech advancements"
        backstory: str = "You are a great at creating insightful articles."
        verbose: bool = True
        allow_delegation: bool = True
        max_iter: int = 25
        max_rpm: None = None
        tools: list[dict[str, Any]] = Field(default_factory=list)
        id: uuid.UUID = Field(default_factory=uuid.uuid4)
        key: str = "agent-key"
        cache: bool = True
        crew: Any = None

    class _CrewLike:
        def __init__(self, agent: Any) -> None:
            self.agent = agent

    agent = _AgentLike()
    crew = _CrewLike(agent)
    agent.crew = crew

    input_value = _get_input_value(Task._execute_core, agent, None, None)
    payload = json.loads(input_value)
    assert isinstance(payload, dict)

    assert payload["context"] is None
    assert payload["tools"] is None
    assert isinstance(payload["agent"], dict)
    _assert_serialized_agent_payload(
        payload["agent"],
        role="Tech Content Strategist",
        goal="Craft compelling content on tech advancements",
        backstory="You are a great at creating insightful articles.",
        allow_delegation=True,
        verbose=True,
        max_iter=25,
        tool_names=[],
    )


@pytest.mark.vcr
def test_crewai_instrumentation(in_memory_span_exporter: InMemorySpanExporter) -> None:
    """Verify spans are generated correctly for CrewAI Crews, Agents, Tasks & Flows."""
    kickoff_crew()

    spans = in_memory_span_exporter.get_finished_spans()
    expected_spans = 4
    assert len(spans) == expected_spans, f"Expected {expected_spans} spans, got {len(spans)}"

    crew_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    assert len(crew_spans) == 1
    crew_span = crew_spans[0]

    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    assert len(agent_spans) == 2

    tool_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.TOOL.value)
    assert len(tool_spans) == 1
    tool_span = tool_spans[0]

    # Verify Crew CHAIN span
    attributes = dict(crew_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert crew_span.name.endswith(".kickoff")
    kickoff_id = attributes.pop("kickoff_id")
    assert isinstance(kickoff_id, str) and uuid.UUID(kickoff_id)
    assert attributes.pop(INPUT_VALUE)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(OUTPUT_VALUE)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    attributes.pop("crew_key")
    attributes.pop("crew_id")
    attributes.pop("crew_inputs")
    attributes.pop("crew_agents")
    attributes.pop("crew_tasks")
    assert not attributes

    # Verify AGENT spans — split by role for specific value assertions
    scraper_span = next(s for s in agent_spans if "Website Scraper" in s.name)
    analyzer_span = next(s for s in agent_spans if "Content Analyzer" in s.name)

    attributes = dict(scraper_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    assert scraper_span.name == "Website Scraper.scrape-task._execute_core"
    assert attributes.pop(GRAPH_NODE_ID) == "Website Scraper"
    assert attributes.pop("task_name") == "scrape-task"
    _assert_serialized_agent_payload(
        _pop_input_payload(attributes)["agent"],
        role="Website Scraper",
        goal="Scrape content from URL",
        backstory="You extract text from websites",
        allow_delegation=False,
        verbose=True,
        max_iter=2,
        tool_names=["scrape_website"],
    )
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(OUTPUT_VALUE)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    attributes.pop("task_key")
    attributes.pop("task_id")
    attributes.pop("crew_key")
    attributes.pop("crew_id")
    assert not attributes

    attributes = dict(analyzer_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    assert analyzer_span.name == "Content Analyzer.analyze-task._execute_core"
    assert attributes.pop(GRAPH_NODE_ID) == "Content Analyzer"
    assert attributes.pop(SpanAttributes.GRAPH_NODE_PARENT_ID) == "Website Scraper"
    assert attributes.pop("task_name") == "analyze-task"
    _assert_serialized_agent_payload(
        _pop_input_payload(attributes)["agent"],
        role="Content Analyzer",
        goal="Extract quotes from text",
        backstory="You extract quotes from text",
        allow_delegation=False,
        verbose=True,
        max_iter=2,
        tool_names=[],
    )
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(OUTPUT_VALUE)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    attributes.pop("task_key")
    attributes.pop("task_id")
    attributes.pop("crew_key")
    attributes.pop("crew_id")
    assert not attributes

    # Verify TOOL span
    _tool_description = (
        "Tool Name: scrape_website\n"
        "Tool Arguments: {\n"
        '  "properties": {\n'
        '    "url": {\n'
        '      "description": "The website URL to scrape",\n'
        '      "title": "Url",\n'
        '      "type": "string"\n'
        "    }\n"
        "  },\n"
        '  "required": [\n'
        '    "url"\n'
        "  ],\n"
        '  "title": "MockScrapeWebsiteToolSchema",\n'
        '  "type": "object",\n'
        '  "additionalProperties": false\n'
        "}\n"
        "Tool Description: Scrape text content from a website URL"
    )
    _tool_parameters = (
        '{"properties": {"url": {"description": "The website URL to scrape",'
        ' "title": "Url", "type": "string"}}, "required": ["url"],'
        ' "title": "MockScrapeWebsiteToolSchema", "type": "object"}'
    )
    _tool_output = (
        '"The world as we have created it is a process of our thinking.'
        ' It cannot be changed without changing our thinking."'
        " by Albert Einstein"
    )
    attributes = dict(tool_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.TOOL.value
    assert tool_span.name == "scrape_website.run"
    assert attributes.pop(TOOL_NAME) == "scrape_website"
    assert attributes.pop(TOOL_DESCRIPTION) == _tool_description
    assert attributes.pop(TOOL_PARAMETERS) == _tool_parameters
    assert attributes.pop(INPUT_VALUE) == '{"url": "http://quotes.toscrape.com/"}'
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(OUTPUT_VALUE) == _tool_output
    assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop("tool.description_updated") == False  # noqa: E712
    assert attributes.pop("tool.cache_function") == "<lambda>"
    assert attributes.pop("tool.result_as_answer") == False  # noqa: E712
    assert attributes.pop("tool.current_usage_count") == 0
    assert not attributes

    # Clear spans exporter
    in_memory_span_exporter.clear()

    kickoff_flow()
    spans = in_memory_span_exporter.get_finished_spans()
    # kickoff CHAIN span + step_one node span + step_two node span
    assert len(spans) == 3, f"Expected 3 spans (kickoff + 2 node spans), got {len(spans)}"

    flow_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    assert len(flow_spans) == 3
    kickoff_span = next(s for s in flow_spans if s.name.endswith(".kickoff"))
    node_spans = [s for s in flow_spans if not s.name.endswith(".kickoff")]
    assert len(node_spans) == 2

    # Verify Flow kickoff CHAIN span
    attributes = dict(kickoff_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert kickoff_span.name.endswith(".kickoff")
    kickoff_id = attributes.pop("kickoff_id")
    assert isinstance(kickoff_id, str) and uuid.UUID(kickoff_id)
    assert attributes.pop(INPUT_VALUE)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(OUTPUT_VALUE) == "Step Two Received: Step One Output"
    assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
    attributes.pop("flow_id")
    attributes.pop("flow_inputs")
    assert not attributes

    # Verify flow node CHAIN spans
    node_spans_by_name = {s.attributes["flow.node.name"]: s for s in node_spans if s.attributes}

    attributes = dict(node_spans_by_name["step_one"].attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert attributes.pop("flow.node.name") == "step_one"
    assert attributes.pop("flow.node.type") == "start"
    assert attributes.pop(OUTPUT_VALUE) == "Step One Output"
    assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
    assert not attributes

    attributes = dict(node_spans_by_name["step_two"].attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert attributes.pop("flow.node.name") == "step_two"
    assert attributes.pop("flow.node.type") == "listen"
    assert attributes.pop(OUTPUT_VALUE) == "Step Two Received: Step One Output"
    assert attributes.pop(OUTPUT_MIME_TYPE) == "text/plain"
    assert not attributes


@pytest.mark.vcr
def test_crewai_instrumentation_context_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Test that context attributes are passed through to spans."""
    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        },
        tags=["tag-1", "tag-2"],
        prompt_template="test-prompt-template",
        prompt_template_version="v1.0",
        prompt_template_variables={
            "var-1": "value-1",
            "var-2": "value-2",
        },
    ):
        kickoff_crew()

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) > 0, "No spans created"
    for span in spans:
        attributes = dict(span.attributes or {})
        span_kind = attributes.pop(OPENINFERENCE_SPAN_KIND)

        # Consume span-kind-specific attributes before checking context attributes
        if span_kind == OpenInferenceSpanKindValues.CHAIN.value:
            kickoff_id = attributes.pop("kickoff_id")
            assert isinstance(kickoff_id, str) and uuid.UUID(kickoff_id)
            assert attributes.pop(INPUT_VALUE)
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            assert attributes.pop(OUTPUT_VALUE)
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            attributes.pop("crew_key")
            attributes.pop("crew_id")
            attributes.pop("crew_inputs")
            attributes.pop("crew_agents")
            attributes.pop("crew_tasks")
        elif span_kind == OpenInferenceSpanKindValues.AGENT.value:
            assert attributes.pop(INPUT_VALUE)
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            assert attributes.pop(OUTPUT_VALUE)
            assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
            assert attributes.pop(GRAPH_NODE_ID)
            attributes.pop("task_key")
            attributes.pop("task_id")
            attributes.pop("task_name", None)
            attributes.pop("crew_key", None)
            attributes.pop("crew_id", None)
            attributes.pop(SpanAttributes.GRAPH_NODE_PARENT_ID, None)
        elif span_kind == OpenInferenceSpanKindValues.TOOL.value:
            assert attributes.pop(TOOL_NAME)
            assert attributes.pop(TOOL_DESCRIPTION)
            assert attributes.pop(TOOL_PARAMETERS)
            assert attributes.pop(INPUT_VALUE)
            assert attributes.pop(INPUT_MIME_TYPE) == JSON
            assert attributes.pop(OUTPUT_VALUE)
            attributes.pop(OUTPUT_MIME_TYPE, None)  # text/plain for string outputs
            attributes.pop("tool.description_updated", None)
            attributes.pop("tool.cache_function", None)
            attributes.pop("tool.result_as_answer", None)
            attributes.pop("tool.max_usage_count", None)
            attributes.pop("tool.current_usage_count", None)

        # Verify context attributes are present on all spans
        assert attributes.pop(SESSION_ID) == "my-test-session"
        assert attributes.pop(USER_ID) == "my-test-user"
        assert json.loads(str(attributes.pop(METADATA))) == {
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        }
        tags = attributes.pop(TAG_TAGS)
        assert list(tags) == ["tag-1", "tag-2"]  # type: ignore[arg-type]
        assert attributes.pop(LLM_PROMPT_TEMPLATE) == "test-prompt-template"
        assert attributes.pop(LLM_PROMPT_TEMPLATE_VERSION) == "v1.0"
        assert json.loads(str(attributes.pop(LLM_PROMPT_TEMPLATE_VARIABLES))) == {
            "var-1": "value-1",
            "var-2": "value-2",
        }
        assert not attributes


@pytest.mark.vcr
def test_tool_spans_nested_under_agent_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Regression test for Bug 1: TOOL spans must be nested under AGENT spans.

    Root cause: Agent._execute_with_timeout submits _execute_without_timeout to a
    ThreadPoolExecutor without copying contextvars. Fixed by patching the class with
    _ExecuteWithoutTimeoutContextDescriptor on _execute_without_timeout.
    Removing the descriptor patch from __init__.py will cause this test to fail.

    kickoff_crew() uses a scraper agent with max_execution_time=120 so CrewAI
    takes the _execute_with_timeout → executor.submit(...) path (agent/core.py 429-431).
    """
    kickoff_crew()

    spans = in_memory_span_exporter.get_finished_spans()
    span_by_id = {span.context.span_id: span for span in spans}

    tool_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.TOOL.value)
    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    assert len(tool_spans) >= 1, "Expected at least one TOOL span"
    assert len(agent_spans) >= 1, "Expected at least one AGENT span"

    # All spans must be in a single trace — no orphaned root traces
    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1, (
        f"Expected all spans in 1 trace, got {len(trace_ids)}. "
        "TOOL spans are orphaned: _ExecuteWithoutTimeoutContextDescriptor may be missing."
    )

    # The scraper agent is the one with tools
    scraper_span = next(s for s in agent_spans if "Website Scraper" in s.name)
    tool_span = tool_spans[0]

    # TOOL span must be a direct child of the AGENT span
    assert tool_span.parent is not None, "TOOL span must have a parent span"
    assert tool_span.parent.span_id == scraper_span.context.span_id, (
        f"TOOL span parent must be the AGENT span "
        f"(expected {format(scraper_span.context.span_id, '016x')}, "
        f"got {format(tool_span.parent.span_id, '016x')}). "
        "Context is not crossing the ThreadPoolExecutor boundary."
    )

    _ = span_by_id  # used for context; referenced in test_flow_crew_spans_in_same_trace


@pytest.mark.vcr
def test_flow_crew_spans_in_same_trace(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Regression test for Bug 2: Flow CHAIN and Crew CHAIN must be in the same trace.

    Root cause: Flow.kickoff() calls asyncio.run() which resets contextvars in the new
    event loop. Fixed by _FlowKickoffWrapper creating the span before asyncio.run().
    Removing _FlowKickoffWrapper from __init__.py will cause this test to fail.
    """
    kickoff_flow_with_crew()

    spans = in_memory_span_exporter.get_finished_spans()
    span_by_id = {span.context.span_id: span for span in spans}

    # All spans must be in a single trace
    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1, (
        f"Expected all spans in 1 trace, got {len(trace_ids)}. "
        "Flow CHAIN and Crew CHAIN are in separate traces: "
        "_FlowKickoffWrapper may be missing."
    )

    chain_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    assert len(chain_spans) >= 2, (
        f"Expected at least 2 CHAIN spans (Flow + Crew), got {len(chain_spans)}"
    )

    # The root span must be the Flow kickoff
    root_spans = [s for s in spans if s.parent is None or s.parent.span_id not in span_by_id]
    assert len(root_spans) == 1, f"Expected 1 root span, got {len(root_spans)}"
    assert root_spans[0].name.endswith(".kickoff"), (
        f"Root span should be the Flow kickoff span, got: {root_spans[0].name}"
    )

    # The Crew CHAIN span must be a descendant of the Flow root
    crew_span = next(s for s in chain_spans if s.context.span_id != root_spans[0].context.span_id)
    assert crew_span.parent is not None, "Crew CHAIN span must have a parent"
    # Walk up the parent chain from crew_span to verify it reaches the Flow root
    current = crew_span
    found_root = False
    for _ in range(10):  # guard against infinite loop
        if current.parent is None or current.parent.span_id not in span_by_id:
            break
        current = span_by_id[current.parent.span_id]
        if current.context.span_id == root_spans[0].context.span_id:
            found_root = True
            break
    assert found_root, (
        "Crew CHAIN span is not a descendant of the Flow root span. "
        "asyncio.run() may have stripped the OTel context."
    )


@pytest.mark.vcr
def test_nested_flow_gets_its_own_span(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Regression test: nested flows must each get their own CHAIN span.

    Before the fix, _flow_span_in_progress was a plain boolean. Any
    Flow.kickoff_async() called while the outer flow's sync kickoff() was running
    (i.e. while the flag was True) would skip span creation — including nested flows
    with a different flow_id.  The fix stores the outer flow's flow_id and only
    skips when the id matches, so inner flows still create their own spans.
    """

    class InnerFlow(Flow[Any]):  # type: ignore[misc, unused-ignore]
        @start()  # type: ignore[misc, unused-ignore]
        def inner_step(self) -> str:
            return "inner done"

    class OuterFlow(Flow[Any]):  # type: ignore[misc, unused-ignore]
        @start()  # type: ignore[misc, unused-ignore]
        def outer_step(self) -> Any:
            # Call a nested flow synchronously from within a flow step.
            # The step runs in a thread (asyncio.to_thread) so there is no
            # running event loop here — InnerFlow.kickoff() can call asyncio.run()
            # without conflict, matching the real-world nested-flow pattern.
            return InnerFlow().kickoff()

    OuterFlow().kickoff()

    spans = in_memory_span_exporter.get_finished_spans()
    span_by_id = {s.context.span_id: s for s in spans}
    chain_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)

    # Expect 4 CHAIN spans: OuterFlow.kickoff + outer_step node +
    # InnerFlow.kickoff + inner_step node.
    assert len(chain_spans) == 4, (
        f"Expected 4 CHAIN spans (2 kickoff + 2 node spans), got {len(chain_spans)}. "
        "_flow_span_in_progress may be suppressing the nested flow span."
    )

    # All spans must be in one trace — context must propagate into the nested flow.
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1, (
        f"Expected 1 trace, got {len(trace_ids)}. "
        "InnerFlow spans are orphaned — context not propagated into nested flow."
    )

    # Exactly one CHAIN span must be the root.
    root_chains = [s for s in chain_spans if s.parent is None or s.parent.span_id not in span_by_id]
    assert len(root_chains) == 1, f"Expected 1 root CHAIN span, got {len(root_chains)}"
    # All other CHAIN spans must have parents.
    for s in chain_spans:
        if s is not root_chains[0]:
            assert s.parent is not None, f"Non-root CHAIN span '{s.name}' must have a parent"


@pytest.mark.vcr
def test_crewai_instrumentation_with_agent(
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """Verify that Agent.kickoff() outside a Crew creates an AGENT span."""
    result = kickoff_agent()
    assert result is not None

    spans = in_memory_span_exporter.get_finished_spans()
    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    assert len(agent_spans) == 1, f"Expected 1 AGENT span, got {len(agent_spans)}"

    agent_span = agent_spans[0]
    attributes = dict(agent_span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    assert agent_span.name == "Helpful Assistant.kickoff"
    assert attributes.pop(GRAPH_NODE_ID) == "Helpful Assistant"
    assert attributes.pop("agent.goal") == "Answer questions clearly and concisely"
    assert attributes.pop("agent.backstory") == "You are a helpful assistant."
    assert attributes.pop(INPUT_VALUE) == '{"messages": "What is 2+2?"}'
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output = json.loads(str(attributes.pop(OUTPUT_VALUE)))
    assert output["raw"] == "2 + 2 equals 4."
    assert output["agent_role"] == "Helpful Assistant"
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert not attributes


def test_execute_core_span_name_with_none_attributes() -> None:
    """
    Verify span names never contain the literal string 'None' when
    agent.role or task.name is None.
    """
    wrapped = MagicMock()
    wrapped.__name__ = "_execute_core"

    # Verify when agent.role is None
    agent = MagicMock()
    agent.role = None
    instance = MagicMock()
    instance.name = "my-task"
    result = _get_execute_core_span_name(instance, wrapped, agent)
    assert "None" not in result, f"Literal 'None' in span name: {result}"

    # Verify when task.name is None
    agent.role = "Research Analyst"
    instance.name = None
    result = _get_execute_core_span_name(instance, wrapped, agent)
    assert "None" not in result, f"Literal 'None' in span name: {result}"
    assert result == "Research Analyst._execute_core"

    # Verify when both are None
    agent.role = None
    instance.name = None
    result = _get_execute_core_span_name(instance, wrapped, agent)
    assert "None" not in result, f"Literal 'None' in span name: {result}"

    # Verify when both present
    agent.role = "Research Analyst"
    instance.name = "research-task"
    result = _get_execute_core_span_name(instance, wrapped, agent)
    assert result == "Research Analyst.research-task._execute_core"
