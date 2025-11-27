"""Tests for workflow instrumentation."""

import re
from typing import Any, Generator

import pytest
from agno.agent import Agent
from agno.models.openai.chat import OpenAIChat
from agno.workflow.condition import Condition
from agno.workflow.step import Step
from agno.workflow.workflow import Workflow
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.agno import AgnoInstrumentor
from openinference.semconv.trace import SpanAttributes


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


def is_valid_node_id(node_id: str) -> bool:
    """Validate node ID format (16-character hex string)."""
    return bool(re.match(r"^[0-9a-f]{16}$", node_id))


class TestWorkflowInstrumentation:
    """Tests for workflow instrumentation."""

    def test_basic_workflow(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        """Test basic workflow instrumentation without making API calls."""
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"

        # Create a simple agent and workflow
        agent = Agent(
            name="Test Agent",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Be helpful"],
        )

        step = Step(name="test_step", agent=agent)

        workflow = Workflow(
            name="Test Workflow",
            description="A simple test workflow",
            steps=[step],
        )

        # Try to run the workflow (will fail due to no API key, but that's ok)
        # We're only testing that instrumentation creates spans
        try:
            workflow.run(
                input="What is 2+2?",
                user_id="test_user_456",
                session_id="test_session_789",
            )
        except Exception:
            # Expected to fail without real API, but spans should still be created
            pass

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Find workflow span
        workflow_span = None
        step_span = None
        for span in spans:
            attributes = dict(span.attributes or dict())
            if "Workflow" in span.name and "run" in span.name:
                workflow_span = attributes
            elif "test_step" in span.name:
                step_span = attributes

        # Validate workflow span exists
        assert workflow_span is not None, "Workflow span should be found"

        # Validate workflow attributes
        assert workflow_span.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "CHAIN"
        assert workflow_span.get(SpanAttributes.INPUT_VALUE) is not None
        # Note: OUTPUT_VALUE might be None if workflow fails, which is ok for instrumentation test

        # Validate workflow-specific attributes
        assert workflow_span.get("agno.workflow.description") == "A simple test workflow"
        assert workflow_span.get("agno.workflow.steps_count") == 1
        workflow_steps = workflow_span.get("agno.workflow.steps")
        assert workflow_steps is not None
        assert "test_step" in workflow_steps

        # Validate step types
        step_types = workflow_span.get("agno.workflow.step_types")
        assert step_types is not None
        assert "Step" in step_types

        # Validate graph node attributes
        workflow_node_id = workflow_span.get(SpanAttributes.GRAPH_NODE_ID)
        assert workflow_node_id is not None
        assert is_valid_node_id(workflow_node_id)
        assert workflow_span.get(SpanAttributes.GRAPH_NODE_NAME) == "Test Workflow"

        # Validate user_id and session_id from arguments
        assert workflow_span.get(SpanAttributes.USER_ID) == "test_user_456"
        assert workflow_span.get(SpanAttributes.SESSION_ID) == "test_session_789"

        # Validate workflow ID (set after execution, might be None if failed early)
        # Just check it exists as an attribute
        # Note: workflow.id is set after execution, so might not be present if workflow failed

        # Validate step has workflow as parent
        if step_span is not None:
            step_node_id = step_span.get(SpanAttributes.GRAPH_NODE_ID)
            assert step_node_id is not None
            assert is_valid_node_id(step_node_id)
            assert step_span.get(SpanAttributes.GRAPH_NODE_PARENT_ID) == workflow_node_id
            # Ensure step has unique node ID
            assert step_node_id != workflow_node_id

    def test_multi_step_workflow(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        """Test workflow with multiple steps instrumentation."""
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"

        # Create agents
        agent1 = Agent(
            name="Agent 1",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Do step 1"],
        )

        agent2 = Agent(
            name="Agent 2",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Do step 2"],
        )

        # Create steps
        step1 = Step(name="step_1", agent=agent1)
        step2 = Step(name="step_2", agent=agent2)

        # Create workflow
        workflow = Workflow(
            name="Multi Step Workflow",
            description="Workflow with multiple steps",
            steps=[step1, step2],
        )

        # Try to run workflow (will fail without real API, but spans should be created)
        try:
            workflow.run(input="Test input", session_id="multi_session_123")
        except Exception:
            # Expected to fail without real API
            pass

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Find workflow span
        workflow_span = None
        for span in spans:
            attributes = dict(span.attributes or dict())
            if "Multi_Step_Workflow" in span.name and "run" in span.name:
                workflow_span = attributes
                break

        assert workflow_span is not None

        # Validate multiple steps are recorded
        assert workflow_span.get("agno.workflow.steps_count") == 2
        workflow_steps = workflow_span.get("agno.workflow.steps")
        assert workflow_steps is not None
        assert len(workflow_steps) == 2
        assert "step_1" in workflow_steps
        assert "step_2" in workflow_steps

        # Validate all steps are of type "Step"
        step_types = workflow_span.get("agno.workflow.step_types")
        assert step_types is not None
        assert len(step_types) == 2
        assert all(st == "Step" for st in step_types)

        # Validate session_id was captured
        assert workflow_span.get(SpanAttributes.SESSION_ID) == "multi_session_123"

    @pytest.mark.asyncio
    async def test_async_workflow(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        """Test async workflow instrumentation."""
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"

        agent = Agent(
            name="Async Agent",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Be helpful"],
        )

        step = Step(name="async_step", agent=agent)

        workflow = Workflow(
            name="Async Workflow",
            description="Async test workflow",
            steps=[step],
        )

        # Try to run async workflow (will fail without real API, but spans should be created)
        try:
            await workflow.arun(
                input="Async test",
                user_id="async_user",
                session_id="async_session",
            )
        except Exception:
            # Expected to fail without real API
            pass

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Find workflow span
        workflow_span = None
        for span in spans:
            attributes = dict(span.attributes or dict())
            if "Async_Workflow" in span.name and "arun" in span.name:
                workflow_span = attributes
                break

        assert workflow_span is not None
        assert workflow_span.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "CHAIN"
        assert workflow_span.get("agno.workflow.description") == "Async test workflow"
        assert workflow_span.get(SpanAttributes.USER_ID) == "async_user"
        assert workflow_span.get(SpanAttributes.SESSION_ID) == "async_session"

    def test_workflow_graph_hierarchy(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        """Test that workflow creates proper graph hierarchy."""
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"

        # Create a simple workflow
        agent = Agent(
            name="Graph Agent",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Test"],
        )

        step1 = Step(name="graph_step_1", agent=agent)
        step2 = Step(name="graph_step_2", agent=agent)

        workflow = Workflow(
            name="Graph Test Workflow",
            steps=[step1, step2],
        )

        try:
            # Run workflow (may fail due to API, that's ok)
            workflow.run(input="Graph test")
        except Exception:
            pass  # We're just testing instrumentation

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Collect spans
        workflow_span = None
        step_spans = []

        for span in spans:
            attributes = dict(span.attributes or dict())
            if "Graph_Test_Workflow" in span.name:
                workflow_span = attributes
                workflow_node_id = workflow_span.get(SpanAttributes.GRAPH_NODE_ID)
            elif "graph_step" in span.name and "execute" in span.name:
                step_spans.append(attributes)

        # Validate workflow node
        if workflow_span:
            assert workflow_node_id is not None
            assert is_valid_node_id(workflow_node_id)

            # Validate each step has workflow as parent
            for step_span in step_spans:
                step_parent_id = step_span.get(SpanAttributes.GRAPH_NODE_PARENT_ID)
                assert step_parent_id == workflow_node_id, "Step should have workflow as parent"

    def test_workflow_without_optional_params(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        """Test workflow without user_id or session_id."""
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"

        agent = Agent(
            name="Minimal Agent",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Test"],
        )

        step = Step(name="minimal_step", agent=agent)

        workflow = Workflow(
            name="Minimal Workflow",
            steps=[step],
        )

        try:
            # Run without user_id or session_id
            workflow.run(input="Minimal test")
        except Exception:
            pass

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Find workflow span
        workflow_span = None
        for span in spans:
            attributes = dict(span.attributes or dict())
            if "Minimal_Workflow" in span.name:
                workflow_span = attributes
                break

        # Should still create span even without optional params
        if workflow_span:
            assert workflow_span.get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "CHAIN"
            assert workflow_span.get(SpanAttributes.GRAPH_NODE_ID) is not None
            # user_id and session_id should not be present
            assert workflow_span.get(SpanAttributes.USER_ID) is None
            assert workflow_span.get(SpanAttributes.SESSION_ID) is None

    def test_workflow_with_condition_step(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        """Test workflow with Condition step type."""
        import os

        os.environ["OPENAI_API_KEY"] = "fake_key"

        # Create agents
        researcher = Agent(
            name="Researcher",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Research the topic"],
        )

        fact_checker = Agent(
            name="Fact Checker",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Verify facts"],
        )

        writer = Agent(
            name="Writer",
            model=OpenAIChat(id="gpt-4o-mini"),
            instructions=["Write article"],
        )

        # Create condition evaluator
        def needs_fact_checking(step_input: Any) -> bool:
            """Always returns True for testing"""
            return True

        # Create steps
        research_step = Step(name="research", agent=researcher)
        fact_check_step = Step(name="fact_check", agent=fact_checker)
        write_step = Step(name="write", agent=writer)

        # Create workflow with Condition
        workflow = Workflow(
            name="Conditional Workflow",
            description="Research -> Condition(Fact Check) -> Write",
            steps=[
                research_step,
                Condition(
                    name="fact_check_condition",
                    description="Check if fact-checking is needed",
                    evaluator=needs_fact_checking,
                    steps=[fact_check_step],
                ),
                write_step,
            ],
        )

        try:
            workflow.run(
                input="Test topic",
                user_id="condition_user",
                session_id="condition_session",
            )
        except Exception:
            # Expected to fail without real API
            pass

        # Get spans
        spans = in_memory_span_exporter.get_finished_spans()

        # Find workflow span
        workflow_span = None
        for span in spans:
            attributes = dict(span.attributes or dict())
            if "Conditional_Workflow" in span.name and "run" in span.name:
                workflow_span = attributes
                break

        assert workflow_span is not None, "Workflow span should be found"

        # Validate workflow has Condition in step types
        step_types = workflow_span.get("agno.workflow.step_types")
        assert step_types is not None, "Step types should be present"
        assert "Condition" in step_types, "Should include Condition step type"
        assert "Step" in step_types, "Should also include regular Step types"

        # Validate all steps are recorded (including Condition)
        workflow_steps = workflow_span.get("agno.workflow.steps")
        assert workflow_steps is not None, "Workflow steps should be present"
        assert "research" in workflow_steps
        assert "fact_check_condition" in workflow_steps
        assert "write" in workflow_steps

        # Validate step count (3 total: research, condition, write)
        assert workflow_span.get("agno.workflow.steps_count") == 3

        # Validate user_id and session_id
        assert workflow_span.get(SpanAttributes.USER_ID) == "condition_user"
        assert workflow_span.get(SpanAttributes.SESSION_ID) == "condition_session"
