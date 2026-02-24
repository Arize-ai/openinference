"""Tests for module-level wrapping in agno >= 2.5.

These tests verify that the instrumentor correctly wraps module-level functions
in agno.agent._run and agno.team._run modules.
"""

from typing import Iterator, Tuple

import pytest
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno import AgnoInstrumentor


@pytest.fixture
def tracer_provider_with_exporter() -> Iterator[Tuple[TracerProvider, InMemorySpanExporter]]:
    """Create a tracer provider with an in-memory exporter."""
    tracer_provider = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield tracer_provider, exporter
    exporter.clear()


class TestInstrumentorSetup:
    """Tests for instrumentor setup."""

    def test_instrumentor_stores_original_methods(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Test that instrumentor stores original methods for uninstrumentation."""
        tracer_provider, _ = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Should have stored original methods
        assert hasattr(instrumentor, "_original_run_method")
        assert hasattr(instrumentor, "_original_team_run_method")

        instrumentor.uninstrument()


class TestAgentSpanCreation:
    """Tests for Agent span creation with module-level wrapping."""

    @pytest.fixture
    def instrumented_agent(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> Iterator[Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]]:
        """Create an instrumented agent."""
        tracer_provider, exporter = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        agent = Agent(
            name="Test Agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
            instructions="You are a test agent.",
        )

        yield agent, exporter, instrumentor

        instrumentor.uninstrument()

    def test_agent_run_creates_span(
        self, instrumented_agent: Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that agent.run() creates a span regardless of agno version."""
        agent, exporter, _ = instrumented_agent

        try:
            agent.run("Test message")
        except Exception:
            # Expected to fail without real API key
            pass

        spans = exporter.get_finished_spans()

        # Should have at least one span (the agent span)
        assert len(spans) >= 1, "Expected at least one span to be created"

        # Find the agent span
        agent_spans = [s for s in spans if "Test_Agent" in s.name or "Agent" in s.name]
        assert len(agent_spans) >= 1, "Expected agent span to be created"

        # Verify span attributes
        agent_span = agent_spans[0]
        attributes = dict(agent_span.attributes or {})
        assert attributes.get("openinference.span.kind") == "AGENT"

    def test_agent_span_has_correct_name(
        self, instrumented_agent: Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that agent span has the correct name format."""
        agent, exporter, _ = instrumented_agent

        try:
            agent.run("Test message")
        except Exception:
            pass

        spans = exporter.get_finished_spans()
        agent_spans = [s for s in spans if s.name.endswith(".run")]

        assert len(agent_spans) >= 1, "Expected agent span with .run suffix"
        # Name should be AgentName.run (with spaces replaced by underscores)
        assert any("Test_Agent.run" in s.name for s in agent_spans), (
            f"Expected 'Test_Agent.run' span, got: {[s.name for s in spans]}"
        )

    def test_agent_span_has_agent_id(
        self, instrumented_agent: Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that agent span includes agno.agent.id attribute."""
        agent, exporter, _ = instrumented_agent

        try:
            agent.run("Test message")
        except Exception:
            pass

        spans = exporter.get_finished_spans()
        agent_spans = [
            s for s in spans if dict(s.attributes or {}).get("openinference.span.kind") == "AGENT"
        ]

        assert len(agent_spans) >= 1, "Expected agent span"
        attributes = dict(agent_spans[0].attributes or {})
        assert "agno.agent.id" in attributes, "Expected agno.agent.id attribute"


class TestTeamSpanCreation:
    """Tests for Team span creation with module-level wrapping."""

    @pytest.fixture
    def instrumented_team(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> Iterator[Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]]:
        """Create an instrumented team."""
        tracer_provider, exporter = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        agent = Agent(
            name="Member Agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
            instructions="You are a member agent.",
        )

        team = Team(
            name="Test Team",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
            members=[agent],
            instructions="Delegate to the member agent.",
        )

        yield team, exporter, instrumentor

        instrumentor.uninstrument()

    def test_team_run_creates_span(
        self, instrumented_team: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that team.run() creates a span regardless of agno version."""
        team, exporter, _ = instrumented_team

        try:
            team.run("Test message")
        except Exception:
            pass

        spans = exporter.get_finished_spans()

        # Should have at least one span
        assert len(spans) >= 1, "Expected at least one span to be created"

        # Find the team span
        team_spans = [s for s in spans if "Test_Team" in s.name or "Team" in s.name]
        assert len(team_spans) >= 1, "Expected team span to be created"

    def test_team_span_has_team_id(
        self, instrumented_team: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that team span includes agno.team.id attribute."""
        team, exporter, _ = instrumented_team

        try:
            team.run("Test message")
        except Exception:
            pass

        spans = exporter.get_finished_spans()
        team_spans = [s for s in spans if dict(s.attributes or {}).get("agno.team.id") is not None]

        assert len(team_spans) >= 1, "Expected team span with agno.team.id"


class TestUninstrumentation:
    """Tests for proper uninstrumentation."""

    def test_uninstrument_restores_original_methods(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Test that uninstrument properly restores original methods."""
        tracer_provider, exporter = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()

        # Instrument
        instrumentor.instrument(tracer_provider=tracer_provider)

        # Create and run agent
        agent = Agent(
            name="Test Agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
        )
        try:
            agent.run("Test")
        except Exception:
            pass

        spans_before = len(exporter.get_finished_spans())
        assert spans_before >= 1, "Expected spans while instrumented"

        # Uninstrument
        instrumentor.uninstrument()
        exporter.clear()

        # Run again - should not create spans
        try:
            agent.run("Test after uninstrument")
        except Exception:
            pass

        spans_after = exporter.get_finished_spans()
        # After uninstrumentation, no new spans should be created
        assert len(spans_after) == 0, (
            f"Expected no spans after uninstrument, got {len(spans_after)}"
        )

    def test_can_reinstrument_after_uninstrument(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Test that we can instrument again after uninstrumenting."""
        tracer_provider, exporter = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()

        # First instrumentation
        instrumentor.instrument(tracer_provider=tracer_provider)

        agent = Agent(
            name="Test Agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
        )
        try:
            agent.run("First run")
        except Exception:
            pass

        first_run_spans = len(exporter.get_finished_spans())
        assert first_run_spans >= 1

        # Uninstrument
        instrumentor.uninstrument()
        exporter.clear()

        # Re-instrument
        instrumentor.instrument(tracer_provider=tracer_provider)

        try:
            agent.run("Second run after re-instrument")
        except Exception:
            pass

        second_run_spans = len(exporter.get_finished_spans())
        assert second_run_spans >= 1, "Expected spans after re-instrumentation"

        instrumentor.uninstrument()


class TestSpanHierarchy:
    """Tests for proper span hierarchy (parent-child relationships)."""

    @pytest.fixture
    def instrumented_team_with_agent(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> Iterator[Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]]:
        """Create an instrumented team with a member agent."""
        tracer_provider, exporter = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        member_agent = Agent(
            name="Member Agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
            instructions="You are a member agent.",
        )

        team = Team(
            name="Parent Team",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
            members=[member_agent],
            instructions="Delegate to member.",
        )

        yield team, exporter, instrumentor

        instrumentor.uninstrument()

    def test_team_span_is_root(
        self, instrumented_team_with_agent: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that team span has no parent (is root of trace)."""
        team, exporter, _ = instrumented_team_with_agent

        try:
            team.run("Test message")
        except Exception:
            pass

        spans = exporter.get_finished_spans()
        team_spans = [s for s in spans if dict(s.attributes or {}).get("agno.team.id") is not None]

        if team_spans:
            team_span = team_spans[0]
            # Team span should have no valid parent (it's the root)
            assert team_span.parent is None or not team_span.parent.is_valid, (
                "Team span should be root of trace"
            )

    def test_member_agent_has_team_as_parent(
        self, instrumented_team_with_agent: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that member agent span has team span as parent via graph.node.parent_id."""
        team, exporter, _ = instrumented_team_with_agent

        try:
            team.run("Test message")
        except Exception:
            pass

        spans = exporter.get_finished_spans()

        # Find team and agent spans
        team_spans = [s for s in spans if dict(s.attributes or {}).get("agno.team.id") is not None]
        agent_spans = [
            s
            for s in spans
            if dict(s.attributes or {}).get("agno.agent.id") is not None
            and dict(s.attributes or {}).get("agno.team.id") is None
        ]

        if team_spans and agent_spans:
            team_span = team_spans[0]
            agent_span = agent_spans[0]

            team_attrs = dict(team_span.attributes or {})
            agent_attrs = dict(agent_span.attributes or {})

            team_node_id = team_attrs.get("graph.node.id")
            agent_parent_id = agent_attrs.get("graph.node.parent_id")

            # Agent's parent should be the team
            if team_node_id and agent_parent_id:
                assert agent_parent_id == team_node_id, (
                    f"Agent parent ({agent_parent_id}) should match team node ({team_node_id})"
                )


class TestAsyncMethods:
    """Tests for async method wrapping."""

    @pytest.fixture
    def instrumented_agent(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> Iterator[Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]]:
        """Create an instrumented agent."""
        tracer_provider, exporter = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        agent = Agent(
            name="Async Test Agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
        )

        yield agent, exporter, instrumentor

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_agent_arun_creates_span(
        self, instrumented_agent: Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that agent.arun() creates a span."""
        agent, exporter, _ = instrumented_agent

        try:
            await agent.arun("Async test message")  # type: ignore[misc]
        except Exception:
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) >= 1, "Expected at least one span from async run"

        agent_spans = [
            s for s in spans if dict(s.attributes or {}).get("openinference.span.kind") == "AGENT"
        ]
        assert len(agent_spans) >= 1, "Expected agent span from async run"

    @pytest.fixture
    def instrumented_team(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> Iterator[Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]]:
        """Create an instrumented team."""
        tracer_provider, exporter = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        agent = Agent(
            name="Member Agent",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
        )

        team = Team(
            name="Async Test Team",
            model=OpenAIChat(id="gpt-4o-mini", api_key="fake-key-for-testing"),
            members=[agent],
        )

        yield team, exporter, instrumentor

        instrumentor.uninstrument()

    @pytest.mark.asyncio
    async def test_team_arun_creates_span(
        self, instrumented_team: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that team.arun() creates a span."""
        team, exporter, _ = instrumented_team

        try:
            await team.arun("Async team test message")  # type: ignore[misc]
        except Exception:
            pass

        spans = exporter.get_finished_spans()
        assert len(spans) >= 1, "Expected at least one span from async team run"
