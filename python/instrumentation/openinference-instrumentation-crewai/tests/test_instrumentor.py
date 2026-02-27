import json
import os
import uuid
from typing import Any, Mapping, Sequence, Tuple, cast

import pytest
from crewai import LLM, Agent, Crew, Task
from crewai.crews import CrewOutput
from crewai.flow.flow import Flow, listen, start  # type: ignore[import-untyped, unused-ignore]
from crewai.tools import BaseTool  # type: ignore[import-untyped, unused-ignore]
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.types import AttributeValue
from pydantic import BaseModel, Field

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

# Don't record or send telemetry to CrewAI during tests
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["CREWAI_TESTING"] = "true"


class MockScrapeWebsiteToolSchema(BaseModel):
    url: str = Field(..., description="The website URL to scrape")


class MockScrapeWebsiteTool(BaseTool):  # type: ignore[misc, unused-ignore]
    """Mock tool to replace ScrapeWebsiteTool and avoid chromadb dependency."""

    name: str = "scrape_website"
    description: str = "Scrape text content from a website URL"
    args_schema: type[BaseModel] = MockScrapeWebsiteToolSchema

    def _run(self, url: str = "http://quotes.toscrape.com/", **kwargs: Any) -> str:
        """Mock run method that returns simple content."""
        return (
            '"The world as we have created it is a process of our thinking. '
            'It cannot be changed without changing our thinking." by Albert Einstein'
        )


def test_entrypoint_for_opentelemetry_instrument() -> None:
    """Test that the instrumentor is properly registered and implements OITracer."""
    instrumentor_entrypoints = list(
        entry_points(  # type: ignore[no-untyped-call]
            group="opentelemetry_instrumentor",
            name="crewai",
        )
    )
    assert len(instrumentor_entrypoints) == 1
    instrumentor = instrumentor_entrypoints[0].load()()
    assert isinstance(instrumentor, CrewAIInstrumentor)
    assert isinstance(CrewAIInstrumentor()._tracer, OITracer)


@pytest.mark.vcr
def test_crewai_instrumentation(in_memory_span_exporter: InMemorySpanExporter) -> None:
    """Verify spans are generated correctly for CrewAI Crews, Agents, Tasks & Flows."""
    analyze_task, scrape_task = kickoff_crew()

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

    _verify_crew_span(crew_span)

    _verify_agent_span(
        agent_spans[0], agent_spans[0].name, analyze_task.description, analyze_task.name
    )
    _verify_agent_span(
        agent_spans[1], agent_spans[1].name, scrape_task.description, scrape_task.name
    )

    _verify_tool_span(tool_span)

    # Clear spans exporter
    in_memory_span_exporter.clear()

    kickoff_flow()
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, f"Expected 1 span (flow), got {len(spans)}"

    flow_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    assert len(flow_spans) == 1
    flow_span = flow_spans[0]

    _verify_flow_span(flow_span)


def kickoff_crew() -> Tuple[Task, Task]:
    """Initialize a CrewAI setup with a Crew, Agents & Tasks."""
    # API key from environment - only used when re-recording the cassette
    # When using the cassette, the key is not needed
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test")
    url = "http://quotes.toscrape.com/"
    llm = LLM(
        model="gpt-4.1-nano", api_key=openai_api_key, temperature=0
    )  # Use a smaller model for tests

    # Define Agents
    scraper_agent = Agent(
        role="Website Scraper",
        goal="Scrape content from URL",
        backstory="You extract text from websites",
        tools=[MockScrapeWebsiteTool()],
        allow_delegation=False,
        llm=llm,
        max_iter=2,
        max_retry_limit=0,
        max_execution_time=120,  # force ThreadPoolExecutor path (agent/core.py)
        verbose=True,
    )
    analyzer_agent = Agent(
        role="Content Analyzer",
        goal="Extract quotes from text",
        backstory="You extract quotes from text",
        allow_delegation=False,
        llm=llm,
        tools=[],
        max_iter=2,
        max_retry_limit=0,
        verbose=True,
    )

    # Define Tasks
    scrape_task = Task(
        description=f"Call the scrape_website tool to fetch text from {url} and return the result.",
        expected_output="Text content from the website.",
        agent=scraper_agent,
        name="scrape-task",
    )
    analyze_task = Task(
        description="Extract the first quote from the content.",
        expected_output="Quote with author.",
        agent=analyzer_agent,
        context=[scrape_task],
        name="analyze-task",
    )

    # Create Crew
    crew = Crew(
        agents=[scraper_agent, analyzer_agent],
        tasks=[scrape_task, analyze_task],
    )

    # NOTE: We manually set Kickoff ID for testing purposes to simulate the API behavior, that is
    # automatically provided by the CrewAI AMP API when Crews are deployed & executed via REST API.
    test_kickoff_id = str(uuid.uuid4())
    inputs = {"id": test_kickoff_id}
    crew_output = crew.kickoff(inputs=inputs)

    result = ""
    if isinstance(crew_output, CrewOutput):
        result = crew_output.raw
    assert isinstance(result, str)
    expected = "Albert Einstein"
    assert expected in result, "Expected quote not found in result"
    return analyze_task, scrape_task


def kickoff_flow() -> Flow[Any]:
    """Initialize a CrewAI setup with a minimal Flow."""

    class SimpleFlow(Flow[Any]):  # type: ignore[misc, unused-ignore]
        @start()  # type: ignore[misc, unused-ignore]
        def step_one(self) -> str:
            """First step that produces an output."""
            return "Step One Output"

        @listen(step_one)  # type: ignore[misc, unused-ignore]
        def step_two(self, step_one_output: str) -> str:
            """Second step that consumes the output from first step."""
            return f"Step Two Received: {step_one_output}"

    flow = SimpleFlow()

    # NOTE: We manually set Kickoff ID for testing purposes to simulate the API behavior, that is
    # automatically provided by the CrewAI AMP API when Flows are deployed & executed via REST API.
    test_kickoff_id = str(uuid.uuid4())
    inputs = {"id": test_kickoff_id}
    result = flow.kickoff(inputs=inputs)

    assert isinstance(result, str)

    expected = "Step Two Received: Step One Output"
    assert expected in result, "Expected value not found in result"
    return flow


def kickoff_flow_with_crew() -> None:
    """Minimal Flow that calls crew.kickoff() — exercises Bug 2 (Flow→Crew context)."""
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test")
    llm = LLM(model="gpt-4.1-nano", api_key=openai_api_key, temperature=0)

    class CrewFlow(Flow[Any]):  # type: ignore[misc, unused-ignore]
        @start()  # type: ignore[misc, unused-ignore]
        def run_crew(self) -> Any:
            scraper = Agent(
                role="Website Scraper",
                goal="Scrape content from URL",
                backstory="You extract text from websites",
                tools=[MockScrapeWebsiteTool()],
                allow_delegation=False,
                llm=llm,
                max_iter=2,
                max_retry_limit=0,
            )
            task = Task(
                description=(
                    "Call the scrape_website tool to fetch text from "
                    "http://quotes.toscrape.com/ and return the result."
                ),
                expected_output="Text content from the website.",
                agent=scraper,
            )
            crew = Crew(agents=[scraper], tasks=[task])
            return crew.kickoff()

    CrewFlow().kickoff()


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
        _verify_context_attributes(span)


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

    # Expect exactly 2 CHAIN spans: one for OuterFlow, one for InnerFlow.
    assert len(chain_spans) == 2, (
        f"Expected 2 CHAIN spans (OuterFlow + InnerFlow), got {len(chain_spans)}. "
        "_flow_span_in_progress may be suppressing the nested flow span."
    )

    # All spans must be in one trace — context must propagate into the nested flow.
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1, (
        f"Expected 1 trace, got {len(trace_ids)}. "
        "InnerFlow spans are orphaned — context not propagated into nested flow."
    )

    # One CHAIN span must be the root, the other must be nested under it.
    root_chains = [s for s in chain_spans if s.parent is None or s.parent.span_id not in span_by_id]
    assert len(root_chains) == 1, f"Expected 1 root CHAIN span, got {len(root_chains)}"
    nested_chain = next(s for s in chain_spans if s is not root_chains[0])
    assert nested_chain.parent is not None, "Inner flow CHAIN span must have a parent"


def get_spans_by_kind(spans: Sequence[ReadableSpan], kind: str) -> Sequence[ReadableSpan]:
    """Get all spans of a specific OpenInference kind."""
    return sorted(
        [
            span
            for span in spans
            if span.attributes
            and span.attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == kind
        ],
        key=lambda s: s.name,
    )


def _verify_crew_span(span: ReadableSpan) -> None:
    """Verify the CHAIN span for Crew.kickoff has correct attributes."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    # Enhanced naming: expect crew name or fallback pattern
    assert span.name.endswith(".kickoff"), (
        f"Expected span name to end with '.kickoff', got: {span.name}"
    )
    # Verify Kickoff ID
    kickoff_id = attributes.get("kickoff_id")
    assert kickoff_id is not None, "Expected kickoff_id attribute to be present at root level."
    assert isinstance(kickoff_id, str), f"Expected kickoff_id to be string, got {type(kickoff_id)}"
    try:
        uuid.UUID(kickoff_id)
    except ValueError:
        pytest.fail(f"kickoff_id '{kickoff_id}' is not a valid UUID.")


def _verify_flow_span(span: ReadableSpan) -> None:
    """Verify the CHAIN span for Flow.kickoff has correct attributes."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.CHAIN.value
    )
    # Enhanced naming: expect flow name or fallback pattern
    assert span.name.endswith(".kickoff"), (
        f"Expected span name to end with '.kickoff', got: {span.name}"
    )
    # Verify Kickoff ID
    kickoff_id = attributes.get("kickoff_id")
    assert kickoff_id is not None, "Expected kickoff_id attribute to be present at root level."
    assert isinstance(kickoff_id, str), f"Expected kickoff_id to be string, got {type(kickoff_id)}"
    try:
        uuid.UUID(kickoff_id)
    except ValueError:
        pytest.fail(f"kickoff_id '{kickoff_id}' is not a valid UUID.")


def _verify_agent_span(
    span: ReadableSpan,
    expected_name: str,
    expected_task_description: str,
    expected_task_name: str | None = None,
) -> None:
    """Verify an AGENT span has correct attributes."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.AGENT.value
    )
    if expected_task_name is not None:
        assert attributes.get("task_name") == expected_task_name
    # Enhanced naming: expect agent role in span name
    assert span.name.endswith("._execute_core"), (
        f"Expected span name to end with '._execute_core', got: {span.name}"
    )
    # Verify agent role is part of the span name
    graph_node_id = attributes.get(SpanAttributes.GRAPH_NODE_ID)
    if graph_node_id:
        assert str(graph_node_id) in span.name, (
            f"Expected graph node ID '{graph_node_id}' in span name '{span.name}'"
        )
    input_value = attributes.get(SpanAttributes.INPUT_VALUE)
    assert input_value is not None
    assert isinstance(input_value, str)

    # Parse JSON input and verify it has expected structure
    input_data = json.loads(input_value)
    assert list(sorted(input_data.keys())) == ["agent", "context", "tools"]

    output_value = attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert isinstance(output_value, str)


def _verify_tool_span(span: ReadableSpan) -> None:
    """Verify a TOOL span has correct attributes."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    # Verify span kind is TOOL
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.TOOL.value
    ), f"Expected TOOL span kind, got: {attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)}"

    # Verify span name ends with .run (BaseTool.run wrapper)
    assert span.name.endswith(".run"), f"Expected span name to end with '.run', got: {span.name}"

    # Verify tool name is present
    tool_name = attributes.get(SpanAttributes.TOOL_NAME)
    assert tool_name is not None, "Tool span should have a tool name"
    assert isinstance(tool_name, str), f"Tool name should be string, got: {type(tool_name)}"

    # Verify tool name is in span name
    assert str(tool_name) in span.name, (
        f"Expected tool name '{tool_name}' in span name '{span.name}'"
    )

    # Verify tool description is present
    tool_description = attributes.get(SpanAttributes.TOOL_DESCRIPTION)
    assert tool_description is not None, "Tool span should have a description"
    assert isinstance(tool_description, str), (
        f"Tool description should be string, got: {type(tool_description)}"
    )

    # Verify tool parameters (args_schema) is present and valid JSON
    tool_parameters = attributes.get(SpanAttributes.TOOL_PARAMETERS)
    assert tool_parameters is not None, "Tool span should have parameters"
    assert isinstance(tool_parameters, str), (
        f"Tool parameters should be JSON string, got: {type(tool_parameters)}"
    )

    # Verify input value is present
    input_value = attributes.get(SpanAttributes.INPUT_VALUE)
    assert input_value is not None, "Tool span should have input value"
    assert isinstance(input_value, str), f"Input value should be string, got: {type(input_value)}"

    # Verify output value is present
    output_value = attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert output_value is not None, "Tool span should have output value"
    assert isinstance(output_value, str), (
        f"Output value should be string, got: {type(output_value)}"
    )


def _verify_context_attributes(span: ReadableSpan) -> None:
    """Verify that context attributes are present on a span."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.SESSION_ID) == "my-test-session"
    assert attributes.get(SpanAttributes.USER_ID) == "my-test-user"
    assert attributes.get(SpanAttributes.METADATA) == json.dumps(
        {
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {
                "key-1": "val-1",
                "key-2": "val-2",
            },
        }
    )
    tags = attributes.get(SpanAttributes.TAG_TAGS)
    expected_tags = ["tag-1", "tag-2"]
    if isinstance(tags, tuple):
        tags = list(tags)
    assert tags == expected_tags
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE) == "test-prompt-template"
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == "v1.0"
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
        {
            "var-1": "value-1",
            "var-2": "value-2",
        }
    )
