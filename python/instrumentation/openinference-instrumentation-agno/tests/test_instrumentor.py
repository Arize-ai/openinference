from typing import Any, Generator

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
            user_id="test_user_123",
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
            # Validate agent-specific attributes
            assert attributes.get("agno.agent.id") is not None, "Agent ID should be present"
            assert attributes.get("agno.run.id") is not None, "Run ID should be present"
            assert attributes.get("user.id") == "test_user_123"
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


def test_agent_context_propagation_to_llm_spans(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """Test that agent name and ID are propagated to child LLM spans"""
    with test_vcr.use_cassette("agent_run.yaml", filter_headers=["authorization", "X-API-KEY"]):
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"
        agent = Agent(
            name="News Agent",
            model=OpenAIChat(id="gpt-4o-mini"),
            tools=[DuckDuckGoTools()],
            user_id="test_user_123",
        )
        agent.run("What's trending on Twitter?", session_id="test_session")

    spans = in_memory_span_exporter.get_finished_spans()

    # Find agent, LLM, and tool spans
    agent_span = None
    llm_spans = []
    tool_spans = []

    for span in spans:
        attributes = dict(span.attributes or dict())
        span_kind = attributes.get("openinference.span.kind")
        if span_kind == "AGENT":
            agent_span = attributes
        elif span_kind == "LLM":
            llm_spans.append(attributes)
        elif span_kind == "TOOL":
            tool_spans.append(attributes)

    # Verify agent span exists
    assert agent_span is not None, "Agent span should exist"

    # Verify LLM spans have agent context attributes
    assert len(llm_spans) > 0, "At least one LLM span should exist"
    for llm_span in llm_spans:
        assert llm_span.get("agno.agent.name") == "News Agent", (
            f"LLM span should have agent name, got: {llm_span.get('agno.agent.name')}"
        )
        # Agent ID should be present if the agent has an ID
        # (agno.agent.id on LLM span comes from context propagation)

    # Verify tool spans do NOT have agent context attributes
    for tool_span in tool_spans:
        assert tool_span.get("agno.agent.name") is None, (
            "Tool span should NOT have agno.agent.name attribute"
        )
        assert tool_span.get("agno.agent.id") is None, (
            "Tool span should NOT have agno.agent.id attribute (from context propagation)"
        )


def test_agent_context_propagation_unit(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_agno_instrumentation: Any,
) -> None:
    """
    Unit test for agent context propagation without VCR cassettes.
    Tests that context keys are properly set and retrieved.
    """
    from opentelemetry import context as context_api

    from openinference.instrumentation.agno.utils import (
        _AGNO_AGENT_ID_CONTEXT_KEY,
        _AGNO_AGENT_NAME_CONTEXT_KEY,
    )

    # Test that context keys can be set and retrieved
    test_agent_name = "Test Agent"
    test_agent_id = "test-agent-123"

    # Set values in context
    ctx = context_api.get_current()
    ctx = context_api.set_value(_AGNO_AGENT_NAME_CONTEXT_KEY, test_agent_name, ctx)
    ctx = context_api.set_value(_AGNO_AGENT_ID_CONTEXT_KEY, test_agent_id, ctx)

    # Attach and verify
    token = context_api.attach(ctx)
    try:
        retrieved_name = context_api.get_value(_AGNO_AGENT_NAME_CONTEXT_KEY)
        retrieved_id = context_api.get_value(_AGNO_AGENT_ID_CONTEXT_KEY)

        assert retrieved_name == test_agent_name, (
            f"Agent name should be '{test_agent_name}', got '{retrieved_name}'"
        )
        assert retrieved_id == test_agent_id, (
            f"Agent ID should be '{test_agent_id}', got '{retrieved_id}'"
        )
    finally:
        context_api.detach(token)

    # After detach, values should not be present
    assert context_api.get_value(_AGNO_AGENT_NAME_CONTEXT_KEY) is None
    assert context_api.get_value(_AGNO_AGENT_ID_CONTEXT_KEY) is None


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
            tools=[YFinanceTools()],
            instructions="Use tables to display data",
        )

        agent_team = Team(
            name="Team",
            members=[web_agent, finance_agent],
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Always include sources", "Use tables to display data"],
            user_id="team_user_999",
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
    # Validate team-specific attributes
    assert team_span.get("agno.team.id") is not None, "Team ID should be present"
    assert team_span.get("agno.run.id") is not None, "Team run ID should be present"
    assert team_span.get("user.id") == "team_user_999"

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
