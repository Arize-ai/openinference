import json
import os
from typing import Mapping, Sequence, Tuple, cast

import pytest
from crewai import LLM, Agent, Crew, Task
from crewai_tools import ScrapeWebsiteTool  # type: ignore
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


def test_entrypoint_for_opentelemetry_instrument() -> None:
    """Test that the instrumentor is properly registered and implements OITracer."""
    instrumentor_entrypoints = list(entry_points(group="opentelemetry_instrumentor", name="crewai"))
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
    """Test that CrewAI agents and tasks are properly traced."""
    analyze_task, scrape_task = kickoff_crew()

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 4

    crew_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    assert len(crew_spans) == 1
    crew_span = crew_spans[0]

    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    assert len(agent_spans) == 2

    _verify_crew_span(crew_span)

    # Both spans are Task._execute_core since we're sorting by name
    _verify_agent_span(agent_spans[0], "Task._execute_core", scrape_task.description)
    _verify_agent_span(agent_spans[1], "Task._execute_core", analyze_task.description)


def kickoff_crew() -> Tuple[Task, Task]:
    # API key from environment - only used when re-recording the cassette
    # When using the cassette, the key is not needed
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-test")
    url = "http://quotes.toscrape.com/"
    llm = LLM(
        model="gpt-4.1-nano", api_key=openai_api_key, temperature=0
    )  # Use a smaller model for tests
    scraper_agent = Agent(
        role="Website Scraper",
        goal="Scrape the content from a given website URL",
        backstory="You are an expert at extracting text content from websites using scraping tools",
        tools=[ScrapeWebsiteTool()],
        llm=llm,
        max_iterations=1,
        max_retry_limit=0,
    )
    analyzer_agent = Agent(
        role="Content Analyzer",
        goal="Analyze scraped website content to extract useful information",
        backstory=(
            "You are skilled at parsing and summarizing data from text, "
            "such as extracting quotes and authors."
        ),
        llm=llm,
        max_iterations=1,
        max_retry_limit=0,
    )
    scrape_task = Task(
        description=(
            f"Scrape the text content from only the provided URL {url} without following any "
            "links, navigating to other pages, or scraping additional URLs. "
            "Focus on extracting the main body content including quote listings from this single "
            "page."
        ),
        expected_output="A string containing the scraped text from the website.",
        agent=scraper_agent,
    )
    analyze_task = Task(
        description=(
            "From the scraped content, extract a list of quotes and their authors. "
            "Then, identify the first quote and its author. "
            "Format the output as: 'First quote: [quote] by [author]'"
        ),
        expected_output="A string identifying the first quote and its author.",
        agent=analyzer_agent,
        context=[scrape_task],
    )
    crew = Crew(
        agents=[scraper_agent, analyzer_agent],
        tasks=[scrape_task, analyze_task],
    )
    result = crew.kickoff().raw
    assert isinstance(result, str)
    expected = (
        "\u201cThe world as we have created it is a process of our thinking. "
        "It cannot be changed without changing our thinking.\u201d by Albert Einstein"
    )
    assert expected in result, "Expected first quote not found in result"
    return analyze_task, scrape_task


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
    assert span.name == "Crew.kickoff"


def _verify_agent_span(
    span: ReadableSpan, expected_name: str, expected_task_description: str
) -> None:
    """Verify an AGENT span has correct attributes."""
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.AGENT.value
    )
    assert span.name == expected_name
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
