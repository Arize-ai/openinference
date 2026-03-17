import contextvars
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Generator
from unittest.mock import MagicMock

import pytest
from crewai import LLM
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.flow_events import (
    FlowFinishedEvent,
    FlowStartedEvent,
    MethodExecutionFinishedEvent,
    MethodExecutionStartedEvent,
)
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.events.types.task_events import TaskCompletedEvent, TaskStartedEvent
from crewai.events.types.tool_usage_events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.llms.base_llm import llm_call_context
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.crewai import CrewAIInstrumentor
from openinference.instrumentation.crewai._event_listener import OpenInferenceEventListener
from openinference.instrumentation.crewai._wrappers import (
    _ExecuteWithoutTimeoutContextDescriptor,
)
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes

pytestmark = pytest.mark.no_autoinstrument


@pytest.fixture()
def listener(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[OpenInferenceEventListener, None, None]:
    with crewai_event_bus.scoped_handlers():
        listener = OpenInferenceEventListener(tracer_provider=tracer_provider)
        in_memory_span_exporter.clear()
        yield listener
        listener.shutdown()
        in_memory_span_exporter.clear()


@pytest.fixture()
def listener_no_llm(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Generator[OpenInferenceEventListener, None, None]:
    with crewai_event_bus.scoped_handlers():
        listener = OpenInferenceEventListener(
            tracer_provider=tracer_provider,
            create_llm_spans=False,
        )
        in_memory_span_exporter.clear()
        yield listener
        listener.shutdown()
        in_memory_span_exporter.clear()


def _get_spans(exporter: InMemorySpanExporter) -> list[ReadableSpan]:
    crewai_event_bus.flush(timeout=5.0)
    return list(exporter.get_finished_spans())


def _emit_and_flush(source: Any, event: Any) -> None:
    crewai_event_bus.emit(source, event)
    crewai_event_bus.flush(timeout=5.0)


def _mock_crew(name: str = "TestCrew") -> MagicMock:
    crew = MagicMock()
    crew.name = name
    crew.key = "crew-test"
    crew.id = uuid.uuid4()
    crew.agents = []
    crew.tasks = []
    crew.fingerprint = None
    return crew


def _mock_agent(
    role: str = "Researcher",
    goal: str = "Find information",
    backstory: str = "Expert analyst",
) -> MagicMock:
    agent = MagicMock()
    agent.role = role
    agent.goal = goal
    agent.backstory = backstory
    agent.id = uuid.uuid4()
    agent.key = f"agent-{role}"
    agent.crew = None
    agent.fingerprint = None
    return agent


def _mock_task(
    name: str = "research_task",
    description: str = "Research AI observability",
    expected_output: str = "A short report",
) -> MagicMock:
    task = MagicMock()
    task.name = name
    task.description = description
    task.expected_output = expected_output
    task.id = uuid.uuid4()
    task.agent = None
    task.context = None
    task.fingerprint = None
    return task


def _make_agent_started_event(
    agent: Any,
    task: Any,
    task_prompt: str = "Do the work",
    **kwargs: Any,
) -> AgentExecutionStartedEvent:
    return AgentExecutionStartedEvent.model_construct(
        agent=agent,
        task=task,
        tools=kwargs.pop("tools", None),
        task_prompt=task_prompt,
        type="agent_execution_started",
        event_id=str(uuid.uuid4()),
        timestamp=None,
        **kwargs,
    )


def _make_agent_completed_event(
    agent: Any,
    task: Any,
    output: str = "done",
    **kwargs: Any,
) -> AgentExecutionCompletedEvent:
    return AgentExecutionCompletedEvent.model_construct(
        agent=agent,
        task=task,
        output=output,
        type="agent_execution_completed",
        event_id=str(uuid.uuid4()),
        timestamp=None,
        **kwargs,
    )


def _make_task_started_event(task: Any, **kwargs: Any) -> TaskStartedEvent:
    return TaskStartedEvent.model_construct(
        task=task,
        context="Task context",
        type="task_started",
        event_id=str(uuid.uuid4()),
        timestamp=None,
        **kwargs,
    )


def _make_task_completed_event(task: Any, **kwargs: Any) -> TaskCompletedEvent:
    return TaskCompletedEvent.model_construct(
        task=task,
        output="done",
        type="task_completed",
        event_id=str(uuid.uuid4()),
        timestamp=None,
        **kwargs,
    )


class _SlowAgentStartListener(OpenInferenceEventListener):
    def _on_agent_started(self, source: Any, event: AgentExecutionStartedEvent) -> None:
        time.sleep(0.05)
        super()._on_agent_started(source, event)


class _SlowTaskStartListener(OpenInferenceEventListener):
    def _on_task_started(self, source: Any, event: TaskStartedEvent) -> None:
        time.sleep(0.05)
        super()._on_task_started(source, event)


class _SlowToolStartListener(OpenInferenceEventListener):
    def _on_tool_started(self, source: Any, event: ToolUsageStartedEvent) -> None:
        time.sleep(0.05)
        super()._on_tool_started(source, event)


def test_crew_execution_creates_chain_span(
    listener: OpenInferenceEventListener,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    crew = _mock_crew()

    start_event = CrewKickoffStartedEvent(
        crew_name="TestCrew",
        crew=crew,
        inputs={"id": "kickoff-123", "topic": "AI"},
    )
    complete_event = CrewKickoffCompletedEvent(
        crew_name="TestCrew",
        crew=crew,
        output="done",
        total_tokens=7,
        started_event_id=start_event.event_id,
    )

    _emit_and_flush(crew, start_event)
    _emit_and_flush(crew, complete_event)

    spans = _get_spans(in_memory_span_exporter)
    assert len(spans) == 1

    span = spans[0]
    attributes = dict(span.attributes or {})
    assert span.name == "TestCrew.kickoff"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.CHAIN.value
    )
    assert attributes.pop("kickoff_id") == "kickoff-123"
    assert attributes.pop("crew_key") == "crew-test"
    assert attributes.pop("crew_id")
    assert attributes.pop("total_tokens") == 7
    assert attributes.pop(SpanAttributes.INPUT_VALUE)
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "done"


def test_tool_hierarchy_survives_concurrent_start_handler_order(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with crewai_event_bus.scoped_handlers():
        listener = _SlowAgentStartListener(tracer_provider=tracer_provider)
        in_memory_span_exporter.clear()

        crew = _mock_crew()
        agent = _mock_agent()
        task = _mock_task()
        tool_source = MagicMock()

        crew_start = CrewKickoffStartedEvent(crew_name="TestCrew", crew=crew, inputs=None)
        crewai_event_bus.emit(crew, crew_start)
        crewai_event_bus.flush(timeout=5.0)

        agent_start = _make_agent_started_event(
            agent=agent,
            task=task,
            task_prompt="Do research",
            parent_event_id=crew_start.event_id,
        )
        tool_start = ToolUsageStartedEvent(
            tool_name="SearchTool",
            tool_args={"q": "test"},
            parent_event_id=agent_start.event_id,
        )
        tool_finish = ToolUsageFinishedEvent(
            tool_name="SearchTool",
            tool_args={"q": "test"},
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            output="results",
            started_event_id=tool_start.event_id,
            parent_event_id=agent_start.event_id,
        )
        agent_finish = _make_agent_completed_event(
            agent=agent,
            task=task,
            output="Agent output",
            started_event_id=agent_start.event_id,
            parent_event_id=crew_start.event_id,
        )
        crew_finish = CrewKickoffCompletedEvent(
            crew_name="TestCrew",
            crew=crew,
            output="Final output",
            started_event_id=crew_start.event_id,
        )

        crewai_event_bus.emit(agent, agent_start)
        crewai_event_bus.emit(tool_source, tool_start)
        crewai_event_bus.emit(tool_source, tool_finish)
        crewai_event_bus.emit(agent, agent_finish)
        crewai_event_bus.emit(crew, crew_finish)
        crewai_event_bus.flush(timeout=5.0)

        spans = list(in_memory_span_exporter.get_finished_spans())
        assert len(spans) == 3

        crew_span = next(span for span in spans if span.name == "TestCrew.kickoff")
        agent_span = next(span for span in spans if span.name.endswith(".execute"))
        tool_span = next(span for span in spans if span.name == "SearchTool.run")

        assert agent_span.parent is not None
        assert agent_span.parent.span_id == crew_span.context.span_id
        assert tool_span.parent is not None
        assert tool_span.parent.span_id == agent_span.context.span_id
        assert crew_span.context.trace_id == agent_span.context.trace_id
        assert agent_span.context.trace_id == tool_span.context.trace_id

        listener.shutdown()


def test_tool_without_parent_event_id_uses_open_agent_context(
    listener: OpenInferenceEventListener,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    crew = _mock_crew()
    agent = _mock_agent()
    task = _mock_task()
    tool_source = MagicMock()

    crew_start = CrewKickoffStartedEvent(crew_name="TestCrew", crew=crew, inputs=None)
    _emit_and_flush(crew, crew_start)

    agent_start = _make_agent_started_event(
        agent=agent,
        task=task,
        task_prompt="Do research",
        parent_event_id=crew_start.event_id,
    )
    _emit_and_flush(agent, agent_start)

    tool_start = ToolUsageStartedEvent(
        tool_name="SearchTool",
        tool_args={"q": "test"},
        from_agent=agent,
        from_task=task,
    )
    contextvars.Context().run(crewai_event_bus.emit, tool_source, tool_start)
    crewai_event_bus.flush(timeout=5.0)

    tool_finish = ToolUsageFinishedEvent(
        tool_name="SearchTool",
        tool_args={"q": "test"},
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        output="results",
        started_event_id=tool_start.event_id,
        from_agent=agent,
        from_task=task,
    )
    contextvars.Context().run(crewai_event_bus.emit, tool_source, tool_finish)
    crewai_event_bus.flush(timeout=5.0)

    agent_finish = _make_agent_completed_event(
        agent=agent,
        task=task,
        output="Agent output",
        started_event_id=agent_start.event_id,
        parent_event_id=crew_start.event_id,
    )
    _emit_and_flush(agent, agent_finish)

    crew_finish = CrewKickoffCompletedEvent(
        crew_name="TestCrew",
        crew=crew,
        output="Final output",
        started_event_id=crew_start.event_id,
    )
    _emit_and_flush(crew, crew_finish)

    spans = _get_spans(in_memory_span_exporter)
    assert len(spans) == 3

    crew_span = next(span for span in spans if span.name == "TestCrew.kickoff")
    agent_span = next(span for span in spans if span.name.endswith(".execute"))
    tool_span = next(span for span in spans if span.name == "SearchTool.run")

    assert agent_span.parent is not None
    assert agent_span.parent.span_id == crew_span.context.span_id
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == agent_span.context.span_id


def test_reused_agent_rootless_tool_stays_on_current_trace(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with crewai_event_bus.scoped_handlers():
        listener = _SlowAgentStartListener(tracer_provider=tracer_provider)
        in_memory_span_exporter.clear()

        agent = _mock_agent()
        tool_source = MagicMock()

        first_crew = _mock_crew("FirstCrew")
        first_task = _mock_task(name="first_task", description="First task")
        first_crew_start = CrewKickoffStartedEvent(
            crew_name="FirstCrew",
            crew=first_crew,
            inputs=None,
        )
        _emit_and_flush(first_crew, first_crew_start)

        first_task_start = _make_task_started_event(
            task=first_task,
            parent_event_id=first_crew_start.event_id,
        )
        _emit_and_flush(first_task, first_task_start)

        first_agent_start = _make_agent_started_event(
            agent=agent,
            task=first_task,
            task_prompt="Do first research",
            parent_event_id=first_task_start.event_id,
        )
        _emit_and_flush(agent, first_agent_start)

        first_agent_finish = _make_agent_completed_event(
            agent=agent,
            task=first_task,
            output="First agent output",
            started_event_id=first_agent_start.event_id,
            parent_event_id=first_task_start.event_id,
        )
        _emit_and_flush(agent, first_agent_finish)

        first_task_finish = _make_task_completed_event(
            task=first_task,
            started_event_id=first_task_start.event_id,
            parent_event_id=first_crew_start.event_id,
        )
        _emit_and_flush(first_task, first_task_finish)

        first_crew_finish = CrewKickoffCompletedEvent(
            crew_name="FirstCrew",
            crew=first_crew,
            output="First crew output",
            started_event_id=first_crew_start.event_id,
        )
        _emit_and_flush(first_crew, first_crew_finish)

        second_crew = _mock_crew("SecondCrew")
        second_task = _mock_task(name="second_task", description="Second task")
        second_crew_start = CrewKickoffStartedEvent(
            crew_name="SecondCrew",
            crew=second_crew,
            inputs=None,
        )
        _emit_and_flush(second_crew, second_crew_start)

        second_task_start = _make_task_started_event(
            task=second_task,
            parent_event_id=second_crew_start.event_id,
        )
        _emit_and_flush(second_task, second_task_start)

        second_agent_start = _make_agent_started_event(
            agent=agent,
            task=second_task,
            task_prompt="Do second research",
            parent_event_id=second_task_start.event_id,
        )
        second_tool_start = ToolUsageStartedEvent(
            tool_name="SearchTool",
            tool_args={"q": "test"},
            from_agent=agent,
            from_task=second_task,
        )
        second_tool_finish = ToolUsageFinishedEvent(
            tool_name="SearchTool",
            tool_args={"q": "test"},
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            output="Second results",
            started_event_id=second_tool_start.event_id,
            from_agent=agent,
            from_task=second_task,
        )
        second_agent_finish = _make_agent_completed_event(
            agent=agent,
            task=second_task,
            output="Second agent output",
            started_event_id=second_agent_start.event_id,
            parent_event_id=second_task_start.event_id,
        )
        second_task_finish = _make_task_completed_event(
            task=second_task,
            started_event_id=second_task_start.event_id,
            parent_event_id=second_crew_start.event_id,
        )
        second_crew_finish = CrewKickoffCompletedEvent(
            crew_name="SecondCrew",
            crew=second_crew,
            output="Second crew output",
            started_event_id=second_crew_start.event_id,
        )

        crewai_event_bus.emit(agent, second_agent_start)
        contextvars.Context().run(crewai_event_bus.emit, tool_source, second_tool_start)
        contextvars.Context().run(crewai_event_bus.emit, tool_source, second_tool_finish)
        crewai_event_bus.emit(agent, second_agent_finish)
        crewai_event_bus.emit(second_task, second_task_finish)
        crewai_event_bus.emit(second_crew, second_crew_finish)
        crewai_event_bus.flush(timeout=5.0)

        spans = list(in_memory_span_exporter.get_finished_spans())
        assert len(spans) == 5

        first_crew_span = next(span for span in spans if span.name == "FirstCrew.kickoff")
        second_crew_span = next(span for span in spans if span.name == "SecondCrew.kickoff")
        tool_span = next(
            span
            for span in spans
            if span.name == "SearchTool.run"
            and span.attributes
            and span.attributes.get(SpanAttributes.OUTPUT_VALUE) == "Second results"
        )

        assert tool_span.context.trace_id != first_crew_span.context.trace_id
        assert tool_span.context.trace_id == second_crew_span.context.trace_id

        listener.shutdown()


def test_agent_hierarchy_survives_transparent_task_scope(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with crewai_event_bus.scoped_handlers():
        listener = _SlowTaskStartListener(tracer_provider=tracer_provider)
        in_memory_span_exporter.clear()

        crew = _mock_crew()
        agent = _mock_agent()
        task = _mock_task()

        crew_start = CrewKickoffStartedEvent(crew_name="TestCrew", crew=crew, inputs=None)
        _emit_and_flush(crew, crew_start)

        task_start = _make_task_started_event(task=task, parent_event_id=crew_start.event_id)
        agent_start = _make_agent_started_event(
            agent=agent,
            task=task,
            task_prompt="Do research",
            parent_event_id=task_start.event_id,
        )
        agent_finish = _make_agent_completed_event(
            agent=agent,
            task=task,
            output="Agent output",
            started_event_id=agent_start.event_id,
            parent_event_id=task_start.event_id,
        )
        task_finish = _make_task_completed_event(
            task=task,
            started_event_id=task_start.event_id,
            parent_event_id=crew_start.event_id,
        )
        crew_finish = CrewKickoffCompletedEvent(
            crew_name="TestCrew",
            crew=crew,
            output="Final output",
            started_event_id=crew_start.event_id,
        )

        crewai_event_bus.emit(task, task_start)
        crewai_event_bus.emit(agent, agent_start)
        crewai_event_bus.emit(agent, agent_finish)
        crewai_event_bus.emit(task, task_finish)
        crewai_event_bus.emit(crew, crew_finish)
        crewai_event_bus.flush(timeout=5.0)

        spans = list(in_memory_span_exporter.get_finished_spans())
        assert len(spans) == 2

        crew_span = next(span for span in spans if span.name == "TestCrew.kickoff")
        agent_span = next(span for span in spans if span.name.endswith(".execute"))
        assert agent_span.parent is not None
        assert agent_span.parent.span_id == crew_span.context.span_id

        listener.shutdown()


def test_tool_from_cache_survives_concurrent_start_handler_order(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    with crewai_event_bus.scoped_handlers():
        listener = _SlowToolStartListener(tracer_provider=tracer_provider)
        in_memory_span_exporter.clear()

        source = MagicMock()
        start_event = ToolUsageStartedEvent(
            tool_name="SearchTool",
            tool_args={"q": "test"},
        )
        finish_event = ToolUsageFinishedEvent(
            tool_name="SearchTool",
            tool_args={"q": "test"},
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            output="results",
            from_cache=True,
            started_event_id=start_event.event_id,
        )

        crewai_event_bus.emit(source, start_event)
        crewai_event_bus.emit(source, finish_event)
        crewai_event_bus.flush(timeout=5.0)

        spans = list(in_memory_span_exporter.get_finished_spans())
        assert len(spans) == 1

        attributes = dict(spans[0].attributes or {})
        assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "results"
        assert attributes.pop("tool.from_cache") is True

        listener.shutdown()


def test_agent_span_sets_graph_parent_id_from_crew_order(
    listener: OpenInferenceEventListener,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    crew = _mock_crew()
    parent_agent = _mock_agent(role="Research Analyst")
    child_agent = _mock_agent(role="Report Writer")
    parent_agent.crew = crew
    child_agent.crew = crew
    crew.agents = [parent_agent, child_agent]

    task = _mock_task(name="report_task", description="Write the report")
    task.agent = child_agent

    crew_start = CrewKickoffStartedEvent(crew_name="TestCrew", crew=crew, inputs=None)
    agent_start = _make_agent_started_event(
        agent=child_agent,
        task=task,
        task_prompt="Write the report",
        parent_event_id=crew_start.event_id,
    )
    agent_finish = _make_agent_completed_event(
        agent=child_agent,
        task=task,
        output="done",
        started_event_id=agent_start.event_id,
        parent_event_id=crew_start.event_id,
    )
    crew_finish = CrewKickoffCompletedEvent(
        crew_name="TestCrew",
        crew=crew,
        output="done",
        started_event_id=crew_start.event_id,
    )

    _emit_and_flush(crew, crew_start)
    _emit_and_flush(child_agent, agent_start)
    _emit_and_flush(child_agent, agent_finish)
    _emit_and_flush(crew, crew_finish)

    spans = _get_spans(in_memory_span_exporter)
    agent_span = next(span for span in spans if span.name.endswith(".execute"))
    attributes = dict(agent_span.attributes or {})

    assert attributes.pop(SpanAttributes.GRAPH_NODE_ID) == "Report Writer"
    assert attributes.pop(SpanAttributes.GRAPH_NODE_PARENT_ID) == "Research Analyst"


def test_llm_call_records_message_and_token_attributes(
    listener: OpenInferenceEventListener,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    source = LLM(model="gpt-4o-mini")
    with llm_call_context() as call_id:
        start_event = LLMCallStartedEvent(
            model="gpt-4o-mini",
            messages="Hello",
            call_id=call_id,
        )
        complete_event = LLMCallCompletedEvent(
            model="gpt-4o-mini",
            messages="Hello",
            response="Hi there",
            call_type=LLMCallType.LLM_CALL,
            call_id=call_id,
            started_event_id=start_event.event_id,
        )

        _emit_and_flush(source, start_event)
        source._track_token_usage_internal(
            {
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            }
        )
        _emit_and_flush(source, complete_event)

    spans = _get_spans(in_memory_span_exporter)
    assert len(spans) == 1

    span = spans[0]
    attributes = dict(span.attributes or {})
    assert span.name == "gpt-4o-mini.llm_call"
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o-mini"
    assert attributes.pop("llm.call_id") == call_id
    assert attributes.pop("llm.call_type") == LLMCallType.LLM_CALL.value
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hello"
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role") == "user"
    assert attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.content") == "Hello"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Hi there"
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.role") == "assistant"
    assert attributes.pop(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message.content") == "Hi there"
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 11
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 7
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 18


def test_multiple_llm_calls_record_token_counts_per_span(
    listener: OpenInferenceEventListener,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    source = LLM(model="gpt-4o-mini")
    expected_usage = {
        "First response": (13, 5, 18),
        "Second response": (17, 9, 26),
    }

    for prompt, response, usage in (
        ("First prompt", "First response", expected_usage["First response"]),
        ("Second prompt", "Second response", expected_usage["Second response"]),
    ):
        with llm_call_context() as call_id:
            start_event = LLMCallStartedEvent(
                model="gpt-4o-mini",
                messages=prompt,
                call_id=call_id,
            )
            complete_event = LLMCallCompletedEvent(
                model="gpt-4o-mini",
                messages=prompt,
                response=response,
                call_type=LLMCallType.LLM_CALL,
                call_id=call_id,
                started_event_id=start_event.event_id,
            )

            _emit_and_flush(source, start_event)
            source._track_token_usage_internal(
                {
                    "prompt_tokens": usage[0],
                    "completion_tokens": usage[1],
                    "total_tokens": usage[2],
                }
            )
            _emit_and_flush(source, complete_event)

    spans = _get_spans(in_memory_span_exporter)
    llm_spans = [span for span in spans if span.name == "gpt-4o-mini.llm_call"]
    assert len(llm_spans) == 2

    for span in llm_spans:
        attributes = dict(span.attributes or {})
        output_value = attributes[SpanAttributes.OUTPUT_VALUE]
        prompt_tokens, completion_tokens, total_tokens = expected_usage[output_value]
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == prompt_tokens
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == completion_tokens
        assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == total_tokens


def test_create_llm_spans_false_skips_llm_spans(
    listener_no_llm: OpenInferenceEventListener,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    source = MagicMock()
    start_event = LLMCallStartedEvent(
        model="gpt-4o-mini",
        messages="Hello",
        call_id=str(uuid.uuid4()),
    )
    complete_event = LLMCallCompletedEvent(
        model="gpt-4o-mini",
        messages="Hello",
        response="Hi there",
        call_type=LLMCallType.LLM_CALL,
        call_id=start_event.call_id,
        started_event_id=start_event.event_id,
    )

    _emit_and_flush(source, start_event)
    _emit_and_flush(source, complete_event)

    spans = _get_spans(in_memory_span_exporter)
    assert len(spans) == 0

    _ = listener_no_llm


def test_flow_method_records_node_type_when_available(
    listener: OpenInferenceEventListener,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    class _FlowSource:
        name = "MyFlow"
        flow_id = uuid.uuid4()
        _start_methods = ["step_one"]
        _listeners = {"step_two": "step_one"}
        _routers = {"route_next"}

    flow_source = _FlowSource()

    flow_start = FlowStartedEvent(flow_name="MyFlow", inputs={"topic": "ai"})
    step_one_start = MethodExecutionStartedEvent(
        flow_name="MyFlow",
        method_name="step_one",
        params={"topic": "ai"},
        state={},
        parent_event_id=flow_start.event_id,
    )
    step_one_finish = MethodExecutionFinishedEvent(
        flow_name="MyFlow",
        method_name="step_one",
        result="ok",
        state={},
        started_event_id=step_one_start.event_id,
        parent_event_id=flow_start.event_id,
    )
    step_two_start = MethodExecutionStartedEvent(
        flow_name="MyFlow",
        method_name="step_two",
        params={"previous": "ok"},
        state={},
        parent_event_id=flow_start.event_id,
    )
    step_two_finish = MethodExecutionFinishedEvent(
        flow_name="MyFlow",
        method_name="step_two",
        result="done",
        state={},
        started_event_id=step_two_start.event_id,
        parent_event_id=flow_start.event_id,
    )
    flow_finish = FlowFinishedEvent(
        flow_name="MyFlow",
        result="done",
        state={},
        started_event_id=flow_start.event_id,
    )

    _emit_and_flush(flow_source, flow_start)
    _emit_and_flush(flow_source, step_one_start)
    _emit_and_flush(flow_source, step_one_finish)
    _emit_and_flush(flow_source, step_two_start)
    _emit_and_flush(flow_source, step_two_finish)
    _emit_and_flush(flow_source, flow_finish)

    spans = _get_spans(in_memory_span_exporter)
    step_one_span = next(span for span in spans if span.name == "MyFlow.step_one")
    step_two_span = next(span for span in spans if span.name == "MyFlow.step_two")

    assert step_one_span.attributes is not None
    assert step_one_span.attributes["flow.node.type"] == "start"
    assert step_two_span.attributes is not None
    assert step_two_span.attributes["flow.node.type"] == "listen"


def test_use_event_listener_patches_context_thread_propagation(
    tracer_provider: trace_sdk.TracerProvider,
) -> None:
    from crewai.agent.core import Agent as CrewAgent
    from crewai.agents.crew_agent_executor import CrewAgentExecutor

    instrumentor = CrewAIInstrumentor()
    instrumentor.uninstrument()

    original_execute_without_timeout = CrewAgent.__dict__["_execute_without_timeout"]
    original_execute_single_native_tool_call = CrewAgentExecutor.__dict__[
        "_execute_single_native_tool_call"
    ]

    with crewai_event_bus.scoped_handlers():
        instrumentor.instrument(tracer_provider=tracer_provider, use_event_listener=True)
        try:
            assert isinstance(
                CrewAgent.__dict__["_execute_without_timeout"],
                _ExecuteWithoutTimeoutContextDescriptor,
            )
            assert isinstance(
                CrewAgentExecutor.__dict__["_execute_single_native_tool_call"],
                _ExecuteWithoutTimeoutContextDescriptor,
            )
        finally:
            instrumentor.uninstrument()

    assert CrewAgent.__dict__["_execute_without_timeout"] is original_execute_without_timeout
    assert (
        CrewAgentExecutor.__dict__["_execute_single_native_tool_call"]
        is original_execute_single_native_tool_call
    )


def test_use_event_listener_flag_bootstraps_listener(
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    instrumentor = CrewAIInstrumentor()
    instrumentor.uninstrument()

    with crewai_event_bus.scoped_handlers():
        instrumentor.instrument(tracer_provider=tracer_provider, use_event_listener=True)
        in_memory_span_exporter.clear()

        try:
            assert instrumentor._event_listener is not None

            crew = _mock_crew()
            start_event = CrewKickoffStartedEvent(crew_name="TestCrew", crew=crew, inputs=None)
            complete_event = CrewKickoffCompletedEvent(
                crew_name="TestCrew",
                crew=crew,
                output="done",
                started_event_id=start_event.event_id,
            )

            _emit_and_flush(crew, start_event)
            _emit_and_flush(crew, complete_event)

            spans = _get_spans(in_memory_span_exporter)
            assert len(spans) == 1
            assert spans[0].name == "TestCrew.kickoff"
        finally:
            instrumentor.uninstrument()
