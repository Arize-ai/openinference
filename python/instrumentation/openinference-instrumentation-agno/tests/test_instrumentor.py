from typing import Any, Generator, Iterator, Optional

import pytest
import vcr  # type: ignore
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation import OITracer
from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.semconv.trace import SpanAttributes

test_vcr = vcr.VCR(
    serializer="yaml",
    cassette_library_dir="tests/openinference/instrumentation/agno/fixtures/",
    record_mode="never",
    match_on=["uri", "method"],
)


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_agno_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    AgnoInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AgnoInstrumentor().uninstrument()


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(  # type: ignore[no-untyped-call]
            group="opentelemetry_instrumentor",
            name="agno",
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, AgnoInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self, setup_agno_instrumentation: Any) -> None:
        assert isinstance(AgnoInstrumentor()._tracer, OITracer)


def test_agno_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    with test_vcr.use_cassette("agent_run.yaml", filter_headers=["authorization", "X-API-KEY"]):
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"
        agent = Agent(
            name="News Agent",  # For best results, set a name that will be used by the tracer
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[DuckDuckGoTools()],
        )
        agent.run("What's trending on Twitter?", session_id="test_session")
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 4
    checked_spans = 0
    for span in spans:
        attributes = dict(span.attributes or dict())
        if span.name == "News_Agent.run":
            checked_spans += 1
            assert attributes.get("openinference.span.kind") == "AGENT"
            assert attributes.get("output.value")
            assert attributes.get("session.id") == "test_session"
            # assert that there are no tokens on the kickoff chain so that we do not
            # double count token when a user is also instrumenting with another instrumentor
            # that provides token counts via the spans.
            assert attributes.get("llm.token_count.prompt") is None
            assert attributes.get("llm.token_count.completion") is None
            assert attributes.get("llm.token_count.total") is None
            assert span.status.is_ok
        elif span.name == "ToolUsage._use":
            checked_spans += 1
            assert attributes.get("openinference.span.kind") == "TOOL"
            assert attributes.get("output.value")
            assert attributes.get("tool.name") in (
                "Search the internet with Serper",
                "Ask question to coworker",
                "Delegate work to coworker",
            )
            assert span.status.is_ok
        elif span.name == "Task._execute_core":
            checked_spans += 1
            assert attributes.get("output.value")
            assert attributes["openinference.span.kind"] == "AGENT"
            assert attributes.get("input.value")
            assert span.status.is_ok
        elif span.name == "OpenAIChat.invoke":
            checked_spans += 1
            assert attributes.get("openinference.span.kind") == "LLM"
            assert attributes.get("llm.model_name") == "gpt-4o-mini"
            assert attributes.get("llm.provider") == "OpenAI"
            assert span.status.is_ok
    assert checked_spans >= 3  # We expect at least agent, tool, and LLM spans


def test_agno_team_coordinate_instrumentation(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    with test_vcr.use_cassette(
        "team_coordinate_run.yaml", filter_headers=["authorization", "X-API-KEY"]
    ):
        import os
        import re

        os.environ["OPENAI_API_KEY"] = "fake_key"

        web_agent = Agent(
            name="Web Agent",
            role="Search the web for information",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[DuckDuckGoTools()],
            instructions="Always include sources",
        )

        finance_agent = Agent(
            name="Finance Agent",
            role="Get financial data",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[
                YFinanceTools()  # type: ignore
            ],
            instructions="Use tables to display data",
        )

        agent_team = Team(
            name="Team",
            members=[web_agent, finance_agent],
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Always include sources", "Use tables to display data"],
        )

        agent_team.run(
            "What's the market outlook and financial performance of NVIDIA?",
            session_id="test_session",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) >= 2

    # Collect spans by name for graph relationship validation
    team_span = None
    web_agent_span = None
    finance_agent_span = None

    for span in spans:
        attributes = dict(span.attributes or dict())
        if span.name == "Team.run":
            team_span = attributes
        elif span.name == "Web_Agent.run":
            web_agent_span = attributes
        elif span.name == "Finance_Agent.run":
            finance_agent_span = attributes

    # Helper function to validate node ID format (16-character hex string)
    def is_valid_node_id(node_id: str) -> bool:
        return bool(re.match(r"^[0-9a-f]{16}$", node_id))

    # Validate graph attributes for team span
    assert team_span is not None, "Team span should be found"
    team_node_id = team_span.get(SpanAttributes.GRAPH_NODE_ID)
    assert team_node_id is not None, "Team node ID should be present"
    assert isinstance(team_node_id, str), f"Team node ID should be a string: {team_node_id}"
    assert is_valid_node_id(team_node_id), f"Team node ID should be valid hex: {team_node_id}"
    # Team should have no parent (root node)
    assert team_span.get(SpanAttributes.GRAPH_NODE_PARENT_ID) is None

    # Validate graph attributes for web agent span
    if web_agent_span is not None:
        web_agent_node_id = web_agent_span.get(SpanAttributes.GRAPH_NODE_ID)
        assert web_agent_node_id is not None, "Web agent node ID should be present"
        assert isinstance(web_agent_node_id, str), (
            f"Web agent node ID should be a string: {web_agent_node_id}"
        )
        assert is_valid_node_id(web_agent_node_id), (
            f"Web agent node ID should be valid hex: {web_agent_node_id}"
        )
        assert web_agent_span.get(SpanAttributes.GRAPH_NODE_NAME) == "Web Agent"
        # Web agent should have team as parent
        assert web_agent_span.get(SpanAttributes.GRAPH_NODE_PARENT_ID) == team_node_id
        # Ensure web agent has different node ID than team (uniqueness)
        assert web_agent_node_id != team_node_id, "Web agent should have unique node ID"

    # Validate graph attributes for finance agent span
    if finance_agent_span is not None:
        finance_agent_node_id = finance_agent_span.get(SpanAttributes.GRAPH_NODE_ID)
        assert finance_agent_node_id is not None, "Finance agent node ID should be present"
        assert isinstance(finance_agent_node_id, str), (
            f"Finance agent node ID should be a string: {finance_agent_node_id}"
        )
        assert is_valid_node_id(finance_agent_node_id), (
            f"Finance agent node ID should be valid hex: {finance_agent_node_id}"
        )
        assert finance_agent_span.get(SpanAttributes.GRAPH_NODE_NAME) == "Finance Agent"
        # Finance agent should have team as parent
        assert finance_agent_span.get(SpanAttributes.GRAPH_NODE_PARENT_ID) == team_node_id
        # Ensure finance agent has different node ID than team (uniqueness)
        assert finance_agent_node_id != team_node_id, "Finance agent should have unique node ID"

    # If both agents are present, ensure they have different node IDs
    if web_agent_span is not None and finance_agent_span is not None:
        assert web_agent_node_id != finance_agent_node_id, "Agents should have unique node IDs"

    # At least one agent span should be present to validate parent-child relationship
    assert web_agent_span is not None or finance_agent_span is not None, (
        "At least one agent span should be found"
    )


def test_session_id_streaming_regression(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """Regression test: ensure session_id is properly extracted in streaming methods."""
    from unittest.mock import Mock

    from openinference.instrumentation.agno._wrappers import _RunWrapper

    # Test the session_id extraction logic directly
    mock_tracer = Mock()
    mock_span = Mock()
    mock_context_manager = Mock()
    mock_context_manager.__enter__ = Mock(return_value=mock_span)
    mock_context_manager.__exit__ = Mock(return_value=None)
    mock_tracer.start_as_current_span.return_value = mock_context_manager

    run_wrapper = _RunWrapper(tracer=mock_tracer)

    # Mock agent with get_last_run_output method that raises exception when session_id is None
    mock_agent = Mock()

    def mock_get_last_run_output(session_id: str = None) -> Mock:
        if session_id is None:
            raise Exception("No session_id provided")
        mock_output = Mock()
        mock_output.to_json.return_value = '{"result": "test output"}'
        return mock_output

    mock_agent.get_last_run_output = mock_get_last_run_output
    mock_agent.name = "Test Agent"

    # Mock the internal _run_stream method with proper signature
    def mock_run_stream(message: str, session_id: str = None, **kwargs: Any) -> Iterator[str]:
        yield "chunk1"
        yield "chunk2"

    # Test arguments with session_id
    test_args = ("test message",)
    test_kwargs = {"session_id": "test_session_123"}

    # This should not raise "No session_id provided" exception
    try:
        result = list(
            run_wrapper.run_stream(
                wrapped=mock_run_stream, instance=mock_agent, args=test_args, kwargs=test_kwargs
            )
        )
        # If we reach here, the session_id was properly extracted
        assert len(result) == 2
        assert result == ["chunk1", "chunk2"]
        test_passed = True
    except Exception as e:
        if "No session_id provided" in str(e):
            test_passed = False  # The bug is still present
        else:
            raise  # Some other unexpected exception

    assert test_passed, "The session_id extraction should work properly in streaming methods"
