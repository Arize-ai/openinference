import json
import os
from typing import Any, Mapping, Sequence, Tuple, cast

import pytest
from crewai import LLM, Agent, Crew, Task
from crewai.flow.flow import Flow, listen, start  # type: ignore[import-untyped]
from crewai.tools import BaseTool  # type: ignore[import-untyped]
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

# Don't record or send telemetry to CrewAI during tests
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"


class MockScrapeWebsiteTool(BaseTool):  # type: ignore[misc]
    """Mock tool to replace ScrapeWebsiteTool and avoid chromadb dependency."""

    name: str = "scrape_website"
    description: str = "Scrape text content from a website URL"

    def _run(self, url: str = "http://quotes.toscrape.com/") -> str:
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


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda request: request.headers.clear() or request,
    before_record_response=lambda response: dict(response, headers={}),
)
def test_crewai_instrumentation(in_memory_span_exporter: InMemorySpanExporter) -> None:
    """Verify spans are generated correctly for CrewAI Crews, Agents, Tasks & Flows."""
    analyze_task, scrape_task = kickoff_crew()

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 7, f"Expected 7 spans (2 AgentAction + 2 tool + 2 agent + 1 crew), got {len(spans)}"

    crew_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    assert len(crew_spans) == 1
    crew_span = crew_spans[0]

    agent_action_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.UNKNOWN.value)
    assert len(agent_action_spans) == 2

    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    assert len(agent_spans) == 2

    _verify_crew_span(crew_span)

    # Enhanced naming: spans now include agent roles
    _verify_agent_span(agent_spans[0], agent_spans[0].name, scrape_task.description)
    _verify_agent_span(agent_spans[1], agent_spans[1].name, analyze_task.description)

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
        llm=llm,
        max_iter=1,
        max_retry_limit=0,
        verbose=True,
    )
    analyzer_agent = Agent(
        role="Content Analyzer",
        goal="Extract quotes from text",
        backstory="You extract quotes from text",
        llm=llm,
        max_iter=1,
        max_retry_limit=0,
        verbose=True,
    )

    # Define Tasks
    scrape_task = Task(
        description=f"Use scrape_website tool to get content from {url}.",
        expected_output="Text content from the website.",
        agent=scraper_agent,
    )
    analyze_task = Task(
        description="Extract the first quote from the content.",
        expected_output="Quote with author.",
        agent=analyzer_agent,
        context=[scrape_task],
    )

    # Create Crew
    crew = Crew(
        agents=[scraper_agent, analyzer_agent],
        tasks=[scrape_task, analyze_task],
    )
    result = crew.kickoff().raw
    assert isinstance(result, str)

    expected = "Albert Einstein"
    assert expected in result, "Expected quote not found in result"
    return analyze_task, scrape_task


def kickoff_flow() -> Flow[Any]:
    """Initialize a CrewAI setup with a minimal Flow."""

    class SimpleFlow(Flow[Any]):  # type: ignore[misc]
        @start()  # type: ignore[misc]
        def step_one(self) -> str:
            """First step that produces an output."""
            return "Step One Output"

        @listen(step_one)  # type: ignore[misc]
        def step_two(self, step_one_output: str) -> str:
            """Second step that consumes the output from first step."""
            return f"Step Two Received: {step_one_output}"

    flow = SimpleFlow()
    result = flow.kickoff()
    assert isinstance(result, str)

    expected = "Step Two Received: Step One Output"
    assert expected in result, "Expected value not found in result"
    return flow


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda request: request.headers.clear() or request,
    before_record_response=lambda response: dict(response, headers={}),
)
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


def _verify_agent_span(
    span: ReadableSpan, expected_name: str, expected_task_description: str
) -> None:
    """Verify an AGENT span has correct attributes."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.AGENT.value
    )
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
