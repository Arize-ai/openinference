"""Tests for workflow instrumentation."""

import asyncio
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
from openinference.semconv.trace import (
    OpenInferenceMimeTypeValues,
    SpanAttributes,
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
        agent_span = None
        for span in spans:
            attributes = dict(span.attributes or dict())
            if "Workflow" in span.name and "run" in span.name:
                workflow_span = attributes
            elif "test_step" in span.name:
                step_span = attributes
            elif "Test_Agent.run" in span.name:
                agent_span = attributes

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
        assert isinstance(workflow_steps, (tuple, list))
        assert "test_step" in workflow_steps

        # Validate step types
        step_types = workflow_span.get("agno.workflow.step_types")
        assert step_types is not None
        assert isinstance(step_types, (tuple, list))
        assert "Step" in step_types

        # Validate graph node attributes
        workflow_node_id = workflow_span.get(SpanAttributes.GRAPH_NODE_ID)
        assert workflow_node_id is not None
        assert isinstance(workflow_node_id, str)
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
            assert isinstance(step_node_id, str)
            assert is_valid_node_id(step_node_id)
            assert step_span.get(SpanAttributes.GRAPH_NODE_PARENT_ID) == workflow_node_id
            # Ensure step has unique node ID
            assert step_node_id != workflow_node_id

        assert agent_span is not None, "Agent span should be found"
        assert agent_span.get(SpanAttributes.AGENT_NAME) == "Test Agent"

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
        assert isinstance(workflow_steps, (tuple, list))
        assert len(workflow_steps) == 2
        assert "step_1" in workflow_steps
        assert "step_2" in workflow_steps

        # Validate all steps are of type "Step"
        step_types = workflow_span.get("agno.workflow.step_types")
        assert step_types is not None
        assert isinstance(step_types, (tuple, list))
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
            assert isinstance(workflow_node_id, str)
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
        assert isinstance(step_types, (tuple, list))
        assert "Condition" in step_types, "Should include Condition step type"
        assert "Step" in step_types, "Should also include regular Step types"

        # Validate all steps are recorded (including Condition)
        workflow_steps = workflow_span.get("agno.workflow.steps")
        assert workflow_steps is not None, "Workflow steps should be present"
        assert isinstance(workflow_steps, (tuple, list))
        assert "research" in workflow_steps
        assert "fact_check_condition" in workflow_steps
        assert "write" in workflow_steps

        # Validate step count (3 total: research, condition, write)
        assert workflow_span.get("agno.workflow.steps_count") == 3

        # Validate user_id and session_id
        assert workflow_span.get(SpanAttributes.USER_ID) == "condition_user"
        assert workflow_span.get(SpanAttributes.SESSION_ID) == "condition_session"


def _function_step(step_input: Any) -> Any:
    """Plain-callable step that needs no model — lets the workflow complete in tests."""
    from agno.workflow.types import StepOutput

    return StepOutput(step_name="fn_step", content="ok", success=True)


async def _afunction_step(step_input: Any) -> Any:
    from agno.workflow.types import StepOutput

    return StepOutput(step_name="fn_step", content="ok", success=True)


def _find_workflow_root_span(spans: Any, name_substr: str) -> Any:
    for span in spans:
        if name_substr in span.name:
            return dict(span.attributes or dict())
    return None


class TestWorkflowRunId:
    """Regression tests for issue #8243: workflow root span must carry agno.run.id."""

    def test_sync_workflow_sets_run_id(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        workflow = Workflow(
            name="RunIdSyncWorkflow",
            steps=[Step(name="fn_step", executor=_function_step)],
        )
        workflow.run(input="hello", run_id="RUN-SYNC-1")

        workflow_span = _find_workflow_root_span(
            in_memory_span_exporter.get_finished_spans(),
            "RunIdSyncWorkflow.run",
        )
        assert workflow_span is not None
        assert workflow_span.get("agno.run.id") == "RUN-SYNC-1"

    def test_sync_streaming_workflow_sets_run_id(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        workflow = Workflow(
            name="RunIdSyncStreamWorkflow",
            steps=[Step(name="fn_step", executor=_function_step)],
        )
        for _ in workflow.run(input="hello", run_id="RUN-SYNC-STREAM-1", stream=True):
            pass

        workflow_span = _find_workflow_root_span(
            in_memory_span_exporter.get_finished_spans(),
            "RunIdSyncStreamWorkflow.run",
        )
        assert workflow_span is not None
        assert workflow_span.get("agno.run.id") == "RUN-SYNC-STREAM-1"

    async def test_async_workflow_sets_run_id(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        workflow = Workflow(
            name="RunIdAsyncWorkflow",
            steps=[Step(name="fn_step", executor=_afunction_step)],
        )
        await workflow.arun(input="hello", run_id="RUN-ASYNC-1")

        workflow_span = _find_workflow_root_span(
            in_memory_span_exporter.get_finished_spans(),
            "RunIdAsyncWorkflow.arun",
        )
        assert workflow_span is not None
        assert workflow_span.get("agno.run.id") == "RUN-ASYNC-1"

    async def test_async_streaming_workflow_sets_run_id(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        workflow = Workflow(
            name="RunIdAsyncStreamWorkflow",
            steps=[Step(name="fn_step", executor=_afunction_step)],
        )
        async for _ in workflow.arun(input="hello", run_id="RUN-ASYNC-STREAM-1", stream=True):
            pass

        workflow_span = _find_workflow_root_span(
            in_memory_span_exporter.get_finished_spans(),
            "RunIdAsyncStreamWorkflow.arun",
        )
        assert workflow_span is not None
        assert workflow_span.get("agno.run.id") == "RUN-ASYNC-STREAM-1"


class TestExtractOutputPydanticContent:
    """Tests for _extract_output serialization and mime type."""

    def _make_response(self, content: Any) -> Any:
        class _Response:
            def __init__(self, c: Any) -> None:
                self.content = c

        return _Response(content)

    def test_pydantic_content_serialized_to_json(self) -> None:
        from pydantic import BaseModel

        from openinference.instrumentation.agno._workflow_wrapper import _extract_output

        class _Answer(BaseModel):
            answer: str

        output, mime_type = _extract_output(self._make_response(_Answer(answer="done")))
        assert output == '{"answer":"done"}'
        assert mime_type == JSON

    def test_plain_string_content_unchanged(self) -> None:
        from openinference.instrumentation.agno._workflow_wrapper import _extract_output

        output, mime_type = _extract_output(self._make_response("ok"))
        assert output == "ok"
        assert mime_type == TEXT

    def test_unserializable_content_falls_back_to_str(self) -> None:
        from openinference.instrumentation.agno._workflow_wrapper import _extract_output

        class _Boom:
            def model_dump_json(self) -> str:
                raise ValueError("cannot serialize")

            def __str__(self) -> str:
                return "boom-repr"

        output, mime_type = _extract_output(self._make_response(_Boom()))
        assert output == "boom-repr"
        assert mime_type == TEXT

    def test_response_with_model_dump_json_returns_json_mime(self) -> None:
        """Response without .content but with model_dump_json should also yield JSON mime."""
        from pydantic import BaseModel

        from openinference.instrumentation.agno._workflow_wrapper import _extract_output

        class _PydanticResponse(BaseModel):
            result: str

        output, mime_type = _extract_output(_PydanticResponse(result="42"))
        assert output == '{"result":"42"}'
        assert mime_type == JSON

    def test_none_response_returns_empty_text(self) -> None:
        from openinference.instrumentation.agno._workflow_wrapper import _extract_output

        output, mime_type = _extract_output(None)
        assert output == ""
        assert mime_type == TEXT

    def test_plain_string_response_returns_text_mime(self) -> None:
        from openinference.instrumentation.agno._workflow_wrapper import _extract_output

        output, mime_type = _extract_output("hello")
        assert output == "hello"
        assert mime_type == TEXT


class TestWorkflowBackgroundRuns:
    async def test_background_run_does_not_span_arun(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        workflow = Workflow(
            name="BackgroundWorkflow",
            steps=[Step(name="fn_step", executor=_afunction_step)],
        )
        before_tasks = set(asyncio.all_tasks())
        placeholder = await workflow.arun(
            input="hello",
            run_id="RUN-BG-1",
            background=True,
        )
        assert placeholder is not None

        # Wait for the background tasks that agno scheduled to finish.
        new_tasks = set(asyncio.all_tasks()) - before_tasks - {asyncio.current_task()}
        for task in new_tasks:
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                pytest.fail(
                    "Background workflow task did not complete in time; "
                    "test setup may not match the real agno background-run API"
                )
            except Exception:
                pass

        spans = in_memory_span_exporter.get_finished_spans()

        arun_spans = [s for s in spans if "BackgroundWorkflow.arun" == s.name]
        assert len(arun_spans) == 0, (
            "Workflow.arun should not create a span when background=True; "
            "the real span should come from _aexecute instead"
        )

    async def test_background_run_aexecute_span_has_real_output_and_duration(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        workflow = Workflow(
            name="BackgroundOutputWorkflow",
            steps=[Step(name="fn_step", executor=_afunction_step)],
        )
        before_tasks = set(asyncio.all_tasks())
        await workflow.arun(
            input="hello",
            run_id="RUN-BG-2",
            background=True,
        )

        # Wait for the background tasks that agno scheduled to finish.
        new_tasks = set(asyncio.all_tasks()) - before_tasks - {asyncio.current_task()}
        for task in new_tasks:
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                pytest.fail("Background workflow task did not complete in time")
            except Exception:
                pass

        spans = in_memory_span_exporter.get_finished_spans()

        execute_span = None
        for span in spans:
            if "BackgroundOutputWorkflow._aexecute" in span.name and "stream" not in span.name:
                execute_span = span
                break

        assert execute_span is not None, (
            "Expected a span for Workflow._aexecute carrying the real background-run execution"
        )

        attrs = dict(execute_span.attributes or {})

        # Real output should be captured here.
        output_value = attrs.get(SpanAttributes.OUTPUT_VALUE)
        assert output_value != "None"
        assert output_value not in (None, "")

        captured_run_id = attrs.get("agno.run.id")
        assert captured_run_id is not None
        assert re.match(r"^[0-9a-f-]{36}$", captured_run_id), captured_run_id  # type: ignore[call-overload]

        # Span should have real, non-trivial duration.
        assert execute_span.start_time is not None
        assert execute_span.end_time is not None
        assert execute_span.end_time > execute_span.start_time

    async def test_background_run_steps_parent_to_aexecute_not_orphaned(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        workflow = Workflow(
            name="BackgroundParentWorkflow",
            steps=[Step(name="bg_fn_step", executor=_afunction_step)],
        )
        before_tasks = set(asyncio.all_tasks())
        await workflow.arun(
            input="hello",
            run_id="RUN-BG-3",
            background=True,
        )

        # Wait for the background tasks that agno scheduled to finish.
        new_tasks = set(asyncio.all_tasks()) - before_tasks - {asyncio.current_task()}
        for task in new_tasks:
            try:
                await asyncio.wait_for(task, timeout=5)
            except asyncio.TimeoutError:
                pytest.fail("Background workflow task did not complete in time")
            except Exception:
                pass

        spans = in_memory_span_exporter.get_finished_spans()

        execute_span = None
        step_span = None
        for span in spans:
            attributes = dict(span.attributes or dict())
            if "BackgroundParentWorkflow._aexecute" in span.name and "stream" not in span.name:
                execute_span = attributes
            elif "bg_fn_step" in span.name:
                step_span = attributes

        assert execute_span is not None
        assert step_span is not None, "Step span should exist under the background execution"

        execute_node_id = execute_span.get(SpanAttributes.GRAPH_NODE_ID)
        step_parent_id = step_span.get(SpanAttributes.GRAPH_NODE_PARENT_ID)
        assert step_parent_id == execute_node_id, (
            "Step span should be parented to the _aexecute span, not orphaned "
            "under an already-closed arun span"
        )

    async def test_foreground_run_does_not_double_span(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        workflow = Workflow(
            name="ForegroundNoDoubleSpanWorkflow",
            steps=[Step(name="fg_fn_step", executor=_afunction_step)],
        )
        await workflow.arun(input="hello", run_id="RUN-FG-1")

        spans = in_memory_span_exporter.get_finished_spans()

        arun_spans = [s for s in spans if "ForegroundNoDoubleSpanWorkflow.arun" == s.name]
        execute_spans = [
            s
            for s in spans
            if "ForegroundNoDoubleSpanWorkflow._aexecute" in s.name and "stream" not in s.name
        ]

        assert len(arun_spans) == 1, "Foreground arun should still produce exactly one span"
        assert len(execute_spans) == 0, (
            "_aexecute should not create its own span when already nested "
            "under arun's span (would double-span the foreground path)"
        )

    async def test_concurrent_foreground_and_background_runs_do_not_interfere(
        self,
        tracer_provider: TracerProvider,
        in_memory_span_exporter: InMemorySpanExporter,
        setup_agno_instrumentation: Any,
    ) -> None:
        # Slow executor gives the background run time to start and reach
        # _aexecute while the foreground run is still in flight.
        async def slow_step(step_input: Any) -> Any:
            await asyncio.sleep(0.05)
            from agno.workflow.types import StepOutput

            return StepOutput(step_name="fg_step", content="foreground done", success=True)

        workflow = Workflow(
            name="ConcurrentWorkflow",
            steps=[Step(name="fg_step", executor=slow_step)],
        )

        # Start foreground run as a task so it runs concurrently.
        foreground_task = asyncio.create_task(
            workflow.arun(input="foreground input", run_id="RUN-FG-CONCURRENT")
        )

        # Yield briefly so the foreground task has started and set its context marker.
        await asyncio.sleep(0.01)

        # Start background run on the same workflow instance while the foreground
        # is still in flight. With the instance-attribute approach this would race.
        # With the contextvar approach each task has its own context copy.
        before_bg_tasks = asyncio.all_tasks()
        await workflow.arun(
            input="background input",
            background=True,
        )

        # Wait for background task to finish.
        new_bg_tasks = asyncio.all_tasks() - before_bg_tasks - {asyncio.current_task()}
        for task in new_bg_tasks:
            try:
                await asyncio.wait_for(task, timeout=5)
            except Exception:
                pass

        # Wait for foreground task to finish.
        try:
            await asyncio.wait_for(foreground_task, timeout=5)
        except Exception:
            pass

        spans = in_memory_span_exporter.get_finished_spans()

        arun_spans = [s for s in spans if s.name == "ConcurrentWorkflow.arun"]
        execute_spans = [
            s
            for s in spans
            if "ConcurrentWorkflow._aexecute" in s.name and "stream" not in s.name
        ]

        # Foreground path: exactly one arun span, no _aexecute span.
        assert len(arun_spans) == 1, (
            f"Expected 1 foreground arun span, got {len(arun_spans)}"
        )

        # Background path: exactly one _aexecute span.
        assert len(execute_spans) == 1, (
            f"Expected 1 background _aexecute span, got {len(execute_spans)}. "
            "The foreground run's span marker may have leaked into the background "
            "task via the instance attribute (regression of the contextvar fix)."
        )

        # The _aexecute span should carry a real run_id.
        execute_attrs = dict(execute_spans[0].attributes or {})
        captured_run_id = execute_attrs.get("agno.run.id")
        assert captured_run_id is not None
        assert re.match(r"^[0-9a-f-]{36}$", captured_run_id), (
            f"agno.run.id should be a UUID, got {captured_run_id!r}"
        )


# mime types
TEXT = OpenInferenceMimeTypeValues.TEXT.value
JSON = OpenInferenceMimeTypeValues.JSON.value
