"""Tests for continue-run wrapping in agno >= 2.5.

These tests verify that the instrumentor wraps the module-level continue-run
functions in agno.agent._run and agno.team._run, which resume a run paused for
human-in-the-loop input. Without these wraps, a continued run produces no
agent/team span.
"""

from typing import AsyncIterator, Iterator, Tuple

import pytest
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.run import RunContext
from agno.run.agent import RunOutput
from agno.run.messages import RunMessages
from agno.run.team import TeamRunOutput
from agno.session.agent import AgentSession
from agno.session.team import TeamSession
from agno.team import Team
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno import AgnoInstrumentor

CONTINUE_FUNCTION_NAMES = (
    "_continue_run",
    "_continue_run_stream",
    "_acontinue_run",
    "_acontinue_run_stream",
)


@pytest.fixture
def tracer_provider_with_exporter() -> Iterator[Tuple[TracerProvider, InMemorySpanExporter]]:
    """Create a tracer provider with an in-memory exporter."""
    tracer_provider = TracerProvider()
    exporter = InMemorySpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield tracer_provider, exporter
    exporter.clear()


@pytest.fixture
def instrumented_agent(
    tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter],
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


@pytest.fixture
def instrumented_team(
    tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]]:
    """Create an instrumented team."""
    from agno.team import _run as team_run_module

    tracer_provider, exporter = tracer_provider_with_exporter
    tracer = tracer_provider.get_tracer(__name__)

    def continue_run(team: Team, *args: object, **kwargs: object) -> TeamRunOutput:
        with tracer.start_as_current_span("nested-continue-run"):
            return TeamRunOutput(run_id="test-run", session_id="test-session")

    def continue_run_stream(team: Team, *args: object, **kwargs: object) -> Iterator[TeamRunOutput]:
        with tracer.start_as_current_span("nested-continue-run-stream"):
            yield TeamRunOutput(run_id="test-run", session_id="test-session")

    async def acontinue_run(team: Team, *args: object, **kwargs: object) -> TeamRunOutput:
        with tracer.start_as_current_span("nested-acontinue-run"):
            return TeamRunOutput(run_id="test-run", session_id="test-session")

    async def acontinue_run_stream(
        team: Team, *args: object, **kwargs: object
    ) -> AsyncIterator[TeamRunOutput]:
        with tracer.start_as_current_span("nested-acontinue-run-stream"):
            yield TeamRunOutput(run_id="test-run", session_id="test-session")

    monkeypatch.setattr(team_run_module, "_continue_run", continue_run)
    monkeypatch.setattr(team_run_module, "_continue_run_stream", continue_run_stream)
    monkeypatch.setattr(team_run_module, "_acontinue_run", acontinue_run)
    monkeypatch.setattr(team_run_module, "_acontinue_run_stream", acontinue_run_stream)

    instrumentor = AgnoInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)

    team = Team(name="Test Team", members=[])

    yield team, exporter, instrumentor

    instrumentor.uninstrument()


def assert_nested_span_parented(
    exporter: InMemorySpanExporter, continuation_name: str, nested_name: str
) -> None:
    spans = {span.name: span for span in exporter.get_finished_spans()}
    continuation_span = spans[continuation_name]
    nested_span = spans[nested_name]
    assert nested_span.context.trace_id == continuation_span.context.trace_id
    assert nested_span.parent is not None
    assert nested_span.parent.span_id == continuation_span.context.span_id


class TestContinueRunWrapping:
    """Tests that the continue-run entrypoints are wrapped and restored."""

    def test_continue_functions_are_wrapped(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Test that instrumenting wraps every continue-run function."""
        from agno.agent import _run as agent_run_module
        from agno.team import _run as team_run_module

        tracer_provider, _ = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)

        try:
            for module in (agent_run_module, team_run_module):
                for function_name in CONTINUE_FUNCTION_NAMES:
                    function = getattr(module, function_name)
                    assert hasattr(function, "__wrapped__"), (
                        f"{module.__name__}.{function_name} is not wrapped"
                    )
        finally:
            instrumentor.uninstrument()

    def test_uninstrument_restores_continue_functions(
        self, tracer_provider_with_exporter: Tuple[TracerProvider, InMemorySpanExporter]
    ) -> None:
        """Test that uninstrumenting restores the original continue-run functions."""
        from agno.agent import _run as agent_run_module
        from agno.team import _run as team_run_module

        originals = {
            module.__name__: {name: getattr(module, name) for name in CONTINUE_FUNCTION_NAMES}
            for module in (agent_run_module, team_run_module)
        }

        tracer_provider, _ = tracer_provider_with_exporter
        instrumentor = AgnoInstrumentor()
        instrumentor.instrument(tracer_provider=tracer_provider)
        instrumentor.uninstrument()

        for module in (agent_run_module, team_run_module):
            for function_name, original in originals[module.__name__].items():
                assert getattr(module, function_name) is original, (
                    f"{module.__name__}.{function_name} was not restored"
                )


class TestContinueRunSpanCreation:
    """Tests that continued runs create agent spans."""

    def test_continue_run_creates_span(
        self, instrumented_agent: Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that _continue_run creates a span named after the continuation."""
        from agno.agent import _run as agent_run_module

        agent, exporter, _ = instrumented_agent

        try:
            agent_run_module._continue_run(
                agent,
                run_response=RunOutput(run_id="test-run", session_id="test-session"),
                run_messages=RunMessages(),
                run_context=RunContext(run_id="test-run", session_id="test-session"),
                session=AgentSession(session_id="test-session"),
                tools=[],
            )
        except Exception:
            # Expected to fail without a real model/session; the span is
            # started before the wrapped function runs.
            pass

        spans = exporter.get_finished_spans()
        assert any(s.name == "Test_Agent.continue_run" for s in spans), (
            f"Expected 'Test_Agent.continue_run' span, got: {[s.name for s in spans]}"
        )

    async def test_acontinue_run_creates_span(
        self, instrumented_agent: Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that _acontinue_run creates a span named after the continuation."""
        from agno.agent import _run as agent_run_module

        agent, exporter, _ = instrumented_agent

        try:
            await agent_run_module._acontinue_run(
                agent,
                session_id="test-session",
                run_context=RunContext(run_id="test-run", session_id="test-session"),
                run_response=RunOutput(run_id="test-run", session_id="test-session"),
            )
        except Exception:
            pass

        spans = exporter.get_finished_spans()
        assert any(s.name == "Test_Agent.acontinue_run" for s in spans), (
            f"Expected 'Test_Agent.acontinue_run' span, got: {[s.name for s in spans]}"
        )

    def test_continue_run_span_is_agent_kind(
        self, instrumented_agent: Tuple[Agent, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        """Test that the continuation span carries the AGENT span kind."""
        from agno.agent import _run as agent_run_module

        agent, exporter, _ = instrumented_agent

        try:
            agent_run_module._continue_run(
                agent,
                run_response=RunOutput(run_id="test-run", session_id="test-session"),
                run_messages=RunMessages(),
                run_context=RunContext(run_id="test-run", session_id="test-session"),
                session=AgentSession(session_id="test-session"),
                tools=[],
            )
        except Exception:
            pass

        spans = [s for s in exporter.get_finished_spans() if s.name == "Test_Agent.continue_run"]
        assert len(spans) == 1
        attributes = dict(spans[0].attributes or {})
        assert attributes.get("openinference.span.kind") == "AGENT"
        assert attributes.get("session.id") == "test-session"


class TestTeamContinueRunSpanCreation:
    """Tests that all Team continuation paths create team spans."""

    def test_continue_run_creates_span(
        self, instrumented_team: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        from agno.team import _run as team_run_module

        team, exporter, _ = instrumented_team

        team_run_module._continue_run(
            team,
            run_response=TeamRunOutput(run_id="test-run", session_id="test-session"),
            run_messages=RunMessages(),
            run_context=RunContext(run_id="test-run", session_id="test-session"),
            tools=[],
            session=TeamSession(session_id="test-session"),
        )
        assert_nested_span_parented(exporter, "Test_Team.continue_run", "nested-continue-run")

    def test_continue_run_stream_creates_span(
        self, instrumented_team: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        from agno.team import _run as team_run_module

        team, exporter, _ = instrumented_team

        list(
            team_run_module._continue_run_stream(
                team,
                run_response=TeamRunOutput(run_id="test-run", session_id="test-session"),
                run_messages=RunMessages(),
                run_context=RunContext(run_id="test-run", session_id="test-session"),
                tools=[],
                session=TeamSession(session_id="test-session"),
            )
        )
        assert_nested_span_parented(
            exporter, "Test_Team.continue_run", "nested-continue-run-stream"
        )

    async def test_acontinue_run_creates_span(
        self, instrumented_team: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        from agno.team import _run as team_run_module

        team, exporter, _ = instrumented_team

        await team_run_module._acontinue_run(
            team,
            session_id="test-session",
            run_context=RunContext(run_id="test-run", session_id="test-session"),
            run_response=TeamRunOutput(run_id="test-run", session_id="test-session"),
        )
        assert_nested_span_parented(exporter, "Test_Team.acontinue_run", "nested-acontinue-run")

    async def test_acontinue_run_stream_creates_span(
        self, instrumented_team: Tuple[Team, InMemorySpanExporter, AgnoInstrumentor]
    ) -> None:
        from agno.team import _run as team_run_module

        team, exporter, _ = instrumented_team

        async for _ in team_run_module._acontinue_run_stream(
            team,
            session_id="test-session",
            run_context=RunContext(run_id="test-run", session_id="test-session"),
            run_response=TeamRunOutput(run_id="test-run", session_id="test-session"),
        ):
            pass
        assert_nested_span_parented(
            exporter, "Test_Team.acontinue_run", "nested-acontinue-run-stream"
        )
