from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from _scenarios import kickoff_agent, kickoff_crew, kickoff_flow, kickoff_flow_with_crew
from _span_helpers import (
    GRAPH_NODE_ID,
    INPUT_MIME_TYPE,
    INPUT_VALUE,
    JSON,
    OPENINFERENCE_SPAN_KIND,
    OUTPUT_MIME_TYPE,
    OUTPUT_VALUE,
    TEXT,
    TOOL_NAME,
    get_spans_by_kind,
    pop_prefixed,
)
from crewai.events.event_bus import crewai_event_bus
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import using_attributes
from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

pytestmark = pytest.mark.no_autoinstrument


@pytest.fixture(scope="module")
def vcr_cassette_dir() -> str:
    return str(Path(__file__).with_name("cassettes") / "test_instrumentor")


@pytest.fixture()
def event_listener_instrumented(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    with crewai_event_bus.scoped_handlers():
        instrumentor = CrewAIInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider, use_event_listener=True)
        in_memory_span_exporter.clear()
        try:
            yield
        finally:
            crewai_event_bus.flush(timeout=10.0)
            instrumentor.uninstrument()
            in_memory_span_exporter.clear()


@pytest.fixture()
def event_listener_instrumented_no_llm(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[None, None, None]:
    with crewai_event_bus.scoped_handlers():
        instrumentor = CrewAIInstrumentor()
        instrumentor.instrument(
            tracer_provider=tracer_provider,
            use_event_listener=True,
            create_llm_spans=False,
        )
        in_memory_span_exporter.clear()
        try:
            yield
        finally:
            crewai_event_bus.flush(timeout=10.0)
            instrumentor.uninstrument()
            in_memory_span_exporter.clear()


def _finished_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    crewai_event_bus.flush(timeout=10.0)
    return list(exporter.get_finished_spans())


def _pop_json(attributes: dict[str, Any], key: str) -> Any:
    return json.loads(str(attributes.pop(key)))


def _assert_uuid(value: Any) -> str:
    text = str(value)
    uuid.UUID(text)
    return text


def _span_parent_name(
    span: ReadableSpan,
    spans_by_id: dict[int, ReadableSpan],
) -> str | None:
    if span.parent is None:
        return None
    parent = spans_by_id.get(span.parent.span_id)
    return None if parent is None else parent.name


def _assert_single_trace(spans: list[ReadableSpan]) -> dict[int, ReadableSpan]:
    trace_ids = {span.context.trace_id for span in spans}
    assert len(trace_ids) == 1
    return {span.context.span_id: span for span in spans}


def _assert_crew_span(span: ReadableSpan) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == "crew.kickoff"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    kickoff_id = _assert_uuid(attributes.pop("kickoff_id"))
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert _pop_json(attributes, INPUT_VALUE) == {"id": kickoff_id}
    crew_agents = _pop_json(attributes, "crew_agents")
    assert [agent["role"] for agent in crew_agents] == ["Website Scraper", "Content Analyzer"]
    crew_tasks = _pop_json(attributes, "crew_tasks")
    assert [task["description"] for task in crew_tasks] == [
        (
            "Call the scrape_website tool to fetch text from "
            "http://quotes.toscrape.com/ and return the result."
        ),
        "Extract the first quote from the content.",
    ]
    assert _assert_uuid(attributes.pop("crew_id"))
    assert attributes.pop("crew_key")
    assert attributes.pop("total_tokens") > 0
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    output = _pop_json(attributes, OUTPUT_VALUE)
    assert output["name"] == "analyze-task"
    assert output["agent"] == "Content Analyzer"
    assert output["output_format"] == "raw"
    assert output["raw"].startswith('"The world as we have created it is a process')
    assert len(output["messages"]) == 3
    assert not attributes


def _assert_scraper_agent_span(span: ReadableSpan) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == "Website Scraper.scrape-task.execute"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    assert attributes.pop(GRAPH_NODE_ID) == "Website Scraper"
    assert attributes.pop("agent.goal") == "Scrape content from URL"
    assert attributes.pop("agent.backstory") == "You extract text from websites"
    assert _pop_json(attributes, "agent.tools") == ["scrape_website"]
    assert _assert_uuid(attributes.pop("agent_id"))
    assert attributes.pop("agent_key")
    assert attributes.pop("task_name") == "scrape-task"
    assert _assert_uuid(attributes.pop("task_id"))
    assert attributes.pop("task_description") == (
        "Call the scrape_website tool to fetch text from "
        "http://quotes.toscrape.com/ and return the result."
    )
    assert attributes.pop("task_expected_output") == "Text content from the website."
    assert attributes.pop(INPUT_MIME_TYPE) == TEXT
    assert "Call the scrape_website tool to fetch text from" in attributes.pop(INPUT_VALUE)
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attributes.pop(OUTPUT_VALUE).startswith(
        "The world as we have created it is a process of our thinking."
    )
    assert not attributes


def _assert_analyzer_agent_span(span: ReadableSpan) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == "Content Analyzer.analyze-task.execute"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    assert attributes.pop(GRAPH_NODE_ID) == "Content Analyzer"
    assert attributes.pop(SpanAttributes.GRAPH_NODE_PARENT_ID) == "Website Scraper"
    assert attributes.pop("agent.goal") == "Extract quotes from text"
    assert attributes.pop("agent.backstory") == "You extract quotes from text"
    assert _assert_uuid(attributes.pop("agent_id"))
    assert attributes.pop("agent_key")
    assert attributes.pop("task_name") == "analyze-task"
    assert _assert_uuid(attributes.pop("task_id"))
    assert attributes.pop("task_description") == "Extract the first quote from the content."
    assert attributes.pop("task_expected_output") == "Quote with author."
    assert attributes.pop(INPUT_MIME_TYPE) == TEXT
    input_value = attributes.pop(INPUT_VALUE)
    assert "Extract the first quote from the content." in input_value
    assert "This is the context you're working with:" in input_value
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attributes.pop(OUTPUT_VALUE).startswith('"The world as we have created it is a process')
    assert not attributes


def _assert_tool_span(span: ReadableSpan) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == "scrape_website.run"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.TOOL.value
    assert attributes.pop(TOOL_NAME) == "scrape_website"
    assert attributes.pop("tool.agent_role") == "Website Scraper"
    assert attributes.pop("tool.task_name") == "scrape-task"
    assert attributes.pop("tool.run_attempts") == 0
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert _pop_json(attributes, INPUT_VALUE) == {"url": "http://quotes.toscrape.com/"}
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attributes.pop(OUTPUT_VALUE).endswith("by Albert Einstein")
    assert not attributes


def _assert_llm_span(
    span: ReadableSpan,
    *,
    expected_call_type: str,
    expect_tools: bool,
    expect_output_messages: bool,
    expected_output_mime_type: str,
) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == "gpt-4.1-nano.llm_call"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.LLM.value
    assert attributes.pop("llm.model_name") == "gpt-4.1-nano"
    _assert_uuid(attributes.pop("llm.call_id"))
    assert attributes.pop("llm.call_type") == expected_call_type
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert _pop_json(attributes, INPUT_VALUE)
    input_messages = pop_prefixed(attributes, "llm.input_messages.")
    assert input_messages
    if expect_tools:
        llm_tools = _pop_json(attributes, "llm.tools")
        assert llm_tools[0]["function"]["name"] == "scrape_website"
    else:
        assert attributes.pop("llm.tools", None) is None
    assert attributes.pop("llm.token_count.prompt") > 0
    assert attributes.pop("llm.token_count.completion") > 0
    assert attributes.pop("llm.token_count.total") > 0
    output_messages = pop_prefixed(attributes, "llm.output_messages.")
    if expect_output_messages:
        assert output_messages
    else:
        assert not output_messages
    assert attributes.pop(OUTPUT_MIME_TYPE) == expected_output_mime_type
    if expected_output_mime_type == JSON:
        assert _pop_json(attributes, OUTPUT_VALUE)
    else:
        assert attributes.pop(OUTPUT_VALUE)
    assert attributes.pop("llm.available_functions", None) is None
    assert not attributes


def _assert_flow_kickoff_span(span: ReadableSpan) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == "SimpleFlow.kickoff"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert _assert_uuid(attributes.pop("flow_id"))
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    kickoff_input = _pop_json(attributes, INPUT_VALUE)
    _assert_uuid(kickoff_input["id"])
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attributes.pop(OUTPUT_VALUE) == "Step Two Received: Step One Output"
    assert not attributes


def _assert_flow_node_span(
    span: ReadableSpan,
    *,
    expected_name: str,
    expected_type: str,
    expected_output: str,
    expected_input: dict[str, Any] | None = None,
) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == f"SimpleFlow.{expected_name}"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert attributes.pop("flow.node.name") == expected_name
    assert attributes.pop("flow.node.type") == expected_type
    if expected_input is None:
        assert attributes.pop(INPUT_VALUE, None) is None
        assert attributes.pop(INPUT_MIME_TYPE, None) is None
    else:
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert _pop_json(attributes, INPUT_VALUE) == expected_input
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attributes.pop(OUTPUT_VALUE) == expected_output
    assert not attributes


def _assert_standalone_agent_span(span: ReadableSpan) -> None:
    attributes = dict(span.attributes or {})
    assert span.name == "Helpful Assistant.kickoff"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    assert attributes.pop(GRAPH_NODE_ID) == "Helpful Assistant"
    assert attributes.pop("agent.goal") == "Answer questions clearly and concisely"
    assert attributes.pop("agent.backstory") == "You are a helpful assistant."
    assert _assert_uuid(attributes.pop("agent_id"))
    assert attributes.pop(INPUT_MIME_TYPE) == TEXT
    assert attributes.pop(INPUT_VALUE) == "What is 2+2?"
    assert attributes.pop(OUTPUT_MIME_TYPE) == TEXT
    assert attributes.pop(OUTPUT_VALUE) == "2 + 2 equals 4."
    assert not attributes


@pytest.mark.vcr
@pytest.mark.default_cassette("test_crewai_instrumentation")
def test_event_listener_crewai_instrumentation(
    event_listener_instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    kickoff_crew()

    spans = _finished_spans(in_memory_span_exporter)
    spans_by_id = _assert_single_trace(spans)
    assert len(spans) == 7

    chain_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    tool_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.TOOL.value)
    llm_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.LLM.value)

    assert len(chain_spans) == 1
    assert len(agent_spans) == 2
    assert len(tool_spans) == 1
    assert len(llm_spans) == 3

    crew_span = chain_spans[0]
    scraper_span = next(span for span in agent_spans if span.name.startswith("Website Scraper"))
    analyzer_span = next(span for span in agent_spans if span.name.startswith("Content Analyzer"))
    tool_span = tool_spans[0]

    assert _span_parent_name(crew_span, spans_by_id) is None
    assert _span_parent_name(scraper_span, spans_by_id) == crew_span.name
    assert _span_parent_name(analyzer_span, spans_by_id) == crew_span.name
    assert _span_parent_name(tool_span, spans_by_id) == scraper_span.name

    _assert_crew_span(crew_span)
    _assert_scraper_agent_span(scraper_span)
    _assert_analyzer_agent_span(analyzer_span)
    _assert_tool_span(tool_span)

    scraper_llm_spans = [
        span for span in llm_spans if _span_parent_name(span, spans_by_id) == scraper_span.name
    ]
    analyzer_llm_spans = [
        span for span in llm_spans if _span_parent_name(span, spans_by_id) == analyzer_span.name
    ]
    assert len(scraper_llm_spans) == 2
    assert len(analyzer_llm_spans) == 1

    tool_call_span = next(
        span
        for span in scraper_llm_spans
        if span.attributes and span.attributes["llm.call_type"] == "tool_call"
    )
    scraper_response_span = next(span for span in scraper_llm_spans if span is not tool_call_span)
    analyzer_response_span = analyzer_llm_spans[0]

    _assert_llm_span(
        tool_call_span,
        expected_call_type="tool_call",
        expect_tools=True,
        expect_output_messages=False,
        expected_output_mime_type=JSON,
    )
    _assert_llm_span(
        scraper_response_span,
        expected_call_type="llm_call",
        expect_tools=True,
        expect_output_messages=True,
        expected_output_mime_type=TEXT,
    )
    _assert_llm_span(
        analyzer_response_span,
        expected_call_type="llm_call",
        expect_tools=False,
        expect_output_messages=True,
        expected_output_mime_type=TEXT,
    )

    in_memory_span_exporter.clear()

    kickoff_flow()

    spans = _finished_spans(in_memory_span_exporter)
    spans_by_id = _assert_single_trace(spans)
    assert len(spans) == 3

    flow_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    assert len(flow_spans) == 3
    kickoff_span = next(span for span in flow_spans if span.name.endswith(".kickoff"))
    step_one_span = next(span for span in flow_spans if span.name.endswith(".step_one"))
    step_two_span = next(span for span in flow_spans if span.name.endswith(".step_two"))

    assert _span_parent_name(kickoff_span, spans_by_id) is None
    assert _span_parent_name(step_one_span, spans_by_id) == kickoff_span.name
    assert _span_parent_name(step_two_span, spans_by_id) == kickoff_span.name

    _assert_flow_kickoff_span(kickoff_span)
    _assert_flow_node_span(
        step_one_span,
        expected_name="step_one",
        expected_type="start",
        expected_output="Step One Output",
    )
    _assert_flow_node_span(
        step_two_span,
        expected_name="step_two",
        expected_type="listen",
        expected_output="Step Two Received: Step One Output",
        expected_input={"_0": "Step One Output"},
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_crewai_instrumentation_context_attributes")
def test_event_listener_context_attributes(
    event_listener_instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with using_attributes(
        session_id="my-test-session",
        user_id="my-test-user",
        metadata={
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {"key-1": "val-1", "key-2": "val-2"},
        },
        tags=["tag-1", "tag-2"],
        prompt_template="test-prompt-template",
        prompt_template_version="v1.0",
        prompt_template_variables={"var-1": "value-1", "var-2": "value-2"},
    ):
        kickoff_crew()

    spans = _finished_spans(in_memory_span_exporter)
    assert len(spans) == 7

    for span in spans:
        attributes = dict(span.attributes or {})
        assert attributes[SpanAttributes.SESSION_ID] == "my-test-session"
        assert attributes[SpanAttributes.USER_ID] == "my-test-user"
        assert json.loads(str(attributes[SpanAttributes.METADATA])) == {
            "test-int": 1,
            "test-str": "string",
            "test-list": [1, 2, 3],
            "test-dict": {"key-1": "val-1", "key-2": "val-2"},
        }
        assert list(attributes[SpanAttributes.TAG_TAGS]) == ["tag-1", "tag-2"]  # type: ignore[arg-type]
        assert attributes[SpanAttributes.LLM_PROMPT_TEMPLATE] == "test-prompt-template"
        assert attributes[SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION] == "v1.0"
        assert json.loads(str(attributes[SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES])) == {
            "var-1": "value-1",
            "var-2": "value-2",
        }


@pytest.mark.vcr
@pytest.mark.default_cassette("test_flow_crew_spans_in_same_trace")
def test_event_listener_flow_crew_spans_in_same_trace(
    event_listener_instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    kickoff_flow_with_crew()

    spans = _finished_spans(in_memory_span_exporter)
    spans_by_id = _assert_single_trace(spans)
    assert len(spans) == 7

    chain_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    tool_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.TOOL.value)
    llm_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.LLM.value)

    assert len(chain_spans) == 3
    assert len(agent_spans) == 1
    assert len(tool_spans) == 1
    assert len(llm_spans) == 2

    flow_span = next(span for span in chain_spans if span.name == "CrewFlow.kickoff")
    run_crew_span = next(span for span in chain_spans if span.name == "CrewFlow.run_crew")
    crew_span = next(span for span in chain_spans if span.name == "crew.kickoff")
    agent_span = agent_spans[0]
    tool_span = tool_spans[0]

    assert _span_parent_name(flow_span, spans_by_id) is None
    assert _span_parent_name(run_crew_span, spans_by_id) == flow_span.name
    assert _span_parent_name(crew_span, spans_by_id) == run_crew_span.name
    assert _span_parent_name(agent_span, spans_by_id) == crew_span.name
    assert _span_parent_name(tool_span, spans_by_id) == agent_span.name
    assert all(_span_parent_name(span, spans_by_id) == agent_span.name for span in llm_spans)

    flow_attributes = dict(flow_span.attributes or {})
    assert flow_attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert _assert_uuid(flow_attributes.pop("flow_id"))
    assert flow_attributes.pop(OUTPUT_MIME_TYPE) == JSON
    flow_output = _pop_json(flow_attributes, OUTPUT_VALUE)
    assert flow_output["raw"].startswith("The content from the website is:")
    assert flow_output["tasks_output"][0]["agent"] == "Website Scraper"
    assert not flow_attributes

    method_attributes = dict(run_crew_span.attributes or {})
    assert method_attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert method_attributes.pop("flow.node.name") == "run_crew"
    assert method_attributes.pop("flow.node.type") == "start"
    assert method_attributes.pop(OUTPUT_MIME_TYPE) == JSON
    method_output = _pop_json(method_attributes, OUTPUT_VALUE)
    assert method_output["raw"].startswith("The content from the website is:")
    assert not method_attributes

    crew_attributes = dict(crew_span.attributes or {})
    assert crew_attributes.pop(OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.CHAIN.value
    assert _pop_json(crew_attributes, "crew_agents")[0]["role"] == "Website Scraper"
    assert len(_pop_json(crew_attributes, "crew_tasks")) == 1
    assert _assert_uuid(crew_attributes.pop("crew_id"))
    assert crew_attributes.pop("crew_key")
    assert crew_attributes.pop("total_tokens") > 0
    assert crew_attributes.pop(OUTPUT_MIME_TYPE) == JSON
    crew_output = _pop_json(crew_attributes, OUTPUT_VALUE)
    assert crew_output["agent"] == "Website Scraper"
    assert not crew_attributes


@pytest.mark.vcr
@pytest.mark.default_cassette("test_crewai_instrumentation_with_agent")
def test_event_listener_crewai_instrumentation_with_agent(
    event_listener_instrumented: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    kickoff_agent()

    spans = _finished_spans(in_memory_span_exporter)
    spans_by_id = _assert_single_trace(spans)
    assert len(spans) == 2

    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    llm_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.LLM.value)
    assert len(agent_spans) == 1
    assert len(llm_spans) == 1

    agent_span = agent_spans[0]
    llm_span = llm_spans[0]
    assert _span_parent_name(agent_span, spans_by_id) is None
    assert _span_parent_name(llm_span, spans_by_id) == agent_span.name

    _assert_standalone_agent_span(agent_span)
    _assert_llm_span(
        llm_span,
        expected_call_type="llm_call",
        expect_tools=False,
        expect_output_messages=True,
        expected_output_mime_type=TEXT,
    )


@pytest.mark.vcr
@pytest.mark.default_cassette("test_crewai_instrumentation")
def test_event_listener_create_llm_spans_false_skips_llm_spans(
    event_listener_instrumented_no_llm: None,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    kickoff_crew()

    spans = _finished_spans(in_memory_span_exporter)
    spans_by_id = _assert_single_trace(spans)
    assert len(spans) == 4

    chain_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.CHAIN.value)
    agent_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.AGENT.value)
    tool_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.TOOL.value)
    llm_spans = get_spans_by_kind(spans, OpenInferenceSpanKindValues.LLM.value)

    assert len(chain_spans) == 1
    assert len(agent_spans) == 2
    assert len(tool_spans) == 1
    assert not llm_spans

    crew_span = chain_spans[0]
    scraper_span = next(span for span in agent_spans if span.name.startswith("Website Scraper"))
    analyzer_span = next(span for span in agent_spans if span.name.startswith("Content Analyzer"))
    tool_span = tool_spans[0]

    assert _span_parent_name(crew_span, spans_by_id) is None
    assert _span_parent_name(scraper_span, spans_by_id) == crew_span.name
    assert _span_parent_name(analyzer_span, spans_by_id) == crew_span.name
    assert _span_parent_name(tool_span, spans_by_id) == scraper_span.name
