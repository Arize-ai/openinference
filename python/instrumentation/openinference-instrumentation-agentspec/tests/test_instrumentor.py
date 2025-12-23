from typing import Any, Dict, Generator

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import Resource as OtelSdkResource  # type: ignore
from opentelemetry.sdk.trace import TracerProvider as OtelTracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pyagentspec.agent import Agent
from pyagentspec.flows.edges import ControlFlowEdge
from pyagentspec.flows.flow import Flow
from pyagentspec.flows.nodes import EndNode, LlmNode, StartNode
from pyagentspec.llms import LlmConfig, LlmGenerationConfig, OpenAiConfig
from pyagentspec.property import FloatProperty
from pyagentspec.tools import ServerTool, Tool
from pyagentspec.tracing._basemodel import _PII_MASK
from pyagentspec.tracing.events import (
    AgentExecutionEnd,
    AgentExecutionStart,
    LlmGenerationRequest,
    LlmGenerationResponse,
    ToolExecutionRequest,
    ToolExecutionResponse,
)
from pyagentspec.tracing.events.llmgeneration import ToolCall
from pyagentspec.tracing.messages.message import Message
from pyagentspec.tracing.spans import AgentExecutionSpan, LlmGenerationSpan, ToolExecutionSpan
from pyagentspec.tracing.trace import Trace as AgentSpecTrace
from pyagentspec.tracing.trace import get_trace

from openinference.instrumentation.agentspec import AgentSpecInstrumentor
from openinference.semconv.trace import (
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolCallAttributes,
)


@pytest.fixture(autouse=True)
def ensure_that_tests_are_finally_uninstrumented() -> Generator[None, Any, None]:
    # This fixture ensures that all the tests are uninstrumented at the end,
    # since the Instrumentor is a singleton, so that the failure of a test
    # does not make other tests fail due to a polluted environment
    try:
        yield
    finally:
        if AgentSpecInstrumentor().is_instrumented_by_opentelemetry:
            AgentSpecInstrumentor().uninstrument()


@pytest.fixture
def agentspec_llm_config() -> LlmConfig:
    return OpenAiConfig(name="openai-llm", model_id="non-existent-model")


@pytest.fixture
def agentspec_tool() -> Tool:
    return ServerTool(
        name="add",
        description="Sum two numbers",
        inputs=[FloatProperty(title="x"), FloatProperty(title="y")],
        outputs=[FloatProperty(title="z")],
    )


@pytest.fixture
def agentspec_agent(agentspec_llm_config: LlmConfig) -> Agent:
    return Agent(
        name="assistant",
        llm_config=agentspec_llm_config,
        system_prompt="You are a helpful agent",
    )


@pytest.fixture
def agentspec_flow(agentspec_llm_config: LlmConfig) -> Flow:
    start_node = StartNode(name="start_node")
    end_node = EndNode(name="end_node")
    llm_node = LlmNode(
        name="llm_node", llm_config=agentspec_llm_config, prompt_template="Write a haiku"
    )
    return Flow(
        name="flow",
        start_node=start_node,
        nodes=[start_node, end_node, llm_node],
        control_flow_connections=[
            ControlFlowEdge(name="cf1", from_node=start_node, to_node=llm_node),
            ControlFlowEdge(name="cf2", from_node=llm_node, to_node=end_node),
        ],
    )


def test_instrumentor_triggers_processors_and_exports_agent_spans(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    agentspec_agent: Agent,
) -> None:
    AgentSpecInstrumentor().instrument(skip_dep_check=True, tracer_provider=tracer_provider)

    # Verify trace was created and started
    trace = get_trace()
    assert isinstance(trace, AgentSpecTrace)
    assert len(trace.span_processors) == 1

    # Send a dummy AgentExecutionSpan through the wrapped processor
    # Add a first event having inputs and a last event having outputs,
    # which will be used for span's I/O
    with AgentExecutionSpan(agent=agentspec_agent) as span:
        span.add_event(AgentExecutionStart(agent=agentspec_agent, inputs={"question": "hi"}))
        span.add_event(AgentExecutionEnd(agent=agentspec_agent, outputs={"answer": "hello"}))

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert (
        attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    )
    assert attrs.pop(SpanAttributes.AGENT_NAME) == "assistant"
    assert attrs.pop(SpanAttributes.INPUT_VALUE) == '{"question": "hi"}'
    assert attrs.pop(SpanAttributes.OUTPUT_VALUE) == '{"answer": "hello"}'

    # Cleanup
    AgentSpecInstrumentor().uninstrument()
    trace = get_trace()
    assert trace is None


def test_instrumentor_context_triggers_processors_and_exports_agent_spans(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    agentspec_agent: Agent,
) -> None:
    with AgentSpecInstrumentor().instrument_context(
        skip_dep_check=True, tracer_provider=tracer_provider
    ):
        # Verify trace was created and started
        trace = get_trace()
        assert isinstance(trace, AgentSpecTrace)
        assert len(trace.span_processors) == 1

        # Send a dummy AgentExecutionSpan through the wrapped processor
        # Add a first event having inputs and a last event having outputs,
        # which will be used for span's I/O
        with AgentExecutionSpan(agent=agentspec_agent) as span:
            span.add_event(AgentExecutionStart(agent=agentspec_agent, inputs={"question": "hi"}))
            span.add_event(AgentExecutionEnd(agent=agentspec_agent, outputs={"answer": "hello"}))

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert (
            attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.AGENT.value
        )
        assert attrs.pop(SpanAttributes.AGENT_NAME) == "assistant"
        assert attrs.pop(SpanAttributes.INPUT_VALUE) == '{"question": "hi"}'
        assert attrs.pop(SpanAttributes.OUTPUT_VALUE) == '{"answer": "hello"}'

    trace = get_trace()
    assert trace is None


def test_instrumentor_context_triggers_processors_and_exports_tool_spans(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    agentspec_tool: Tool,
) -> None:
    with AgentSpecInstrumentor().instrument_context(
        skip_dep_check=True, tracer_provider=tracer_provider
    ):
        # Verify trace was created and started
        trace = get_trace()
        assert isinstance(trace, AgentSpecTrace)
        assert len(trace.span_processors) == 1

        with ToolExecutionSpan(tool=agentspec_tool) as span:
            request_id = "abc123"
            span.add_event(
                ToolExecutionRequest(
                    tool=agentspec_tool, request_id=request_id, inputs={"x": 1, "y": 2}
                )
            )
            span.add_event(
                ToolExecutionResponse(tool=agentspec_tool, request_id=request_id, outputs={"z": 3})
            )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert (
            attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.TOOL.value
        )
        assert attrs.pop(SpanAttributes.TOOL_NAME) == agentspec_tool.name
        assert attrs.pop(SpanAttributes.TOOL_DESCRIPTION) == agentspec_tool.description
        # Tool parameters are flattened under tool.parameters.*
        assert attrs.pop("tool.parameters.x") == 1
        assert attrs.pop("tool.parameters.y") == 2
        assert attrs.pop(ToolCallAttributes.TOOL_CALL_ID) == request_id
        assert attrs.pop(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME) == agentspec_tool.name
        assert attrs.pop(ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON) == (
            '{"x": 1, "y": 2}'
        )
        assert attrs.pop(SpanAttributes.INPUT_VALUE) == '{"x": 1, "y": 2}'
        assert attrs.pop(SpanAttributes.OUTPUT_VALUE) == '{"z": 3}'

    trace = get_trace()
    assert trace is None


def test_instrumentor_context_triggers_processors_and_exports_llm_spans(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    agentspec_llm_config: LlmConfig,
    agentspec_tool: Tool,
) -> None:
    with AgentSpecInstrumentor().instrument_context(
        skip_dep_check=True, tracer_provider=tracer_provider
    ):
        # Verify trace was created and started
        trace = get_trace()
        assert isinstance(trace, AgentSpecTrace)
        assert len(trace.span_processors) == 1

        with LlmGenerationSpan(llm_config=agentspec_llm_config) as span:
            request_id = "abc123"
            call_id = "tc123"
            span.add_event(
                LlmGenerationRequest(
                    llm_config=agentspec_llm_config,
                    llm_generation_config=LlmGenerationConfig(temperature=0.3),
                    request_id=request_id,
                    prompt=[Message(role="user", content="1+2")],
                    tools=[agentspec_tool],
                )
            )
            span.add_event(
                LlmGenerationResponse(
                    llm_config=agentspec_llm_config,
                    request_id=request_id,
                    content="",
                    tool_calls=[
                        ToolCall(
                            call_id=call_id,
                            tool_name=agentspec_tool.name,
                            arguments='{"x": 1, "y": 2}',
                        ),
                    ],
                    input_tokens=5,
                    output_tokens=3,
                )
            )

        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes or {})
        assert (
            attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND)
            == OpenInferenceSpanKindValues.LLM.value
        )
        assert attrs.pop(SpanAttributes.LLM_MODEL_NAME) == agentspec_llm_config.model_id
        assert attrs.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS) == '{"temperature": 0.3}'

        # Input messages flattened
        assert attrs.pop("llm.input_messages.0.message.content") == "1+2"

        # Output messages with tool calls flattened
        assert attrs.pop("llm.output_messages.0.message.role") == "assistant"
        assert attrs.pop("llm.output_messages.0.message.tool_calls.0.tool_call.id") == call_id
        assert (
            attrs.pop("llm.output_messages.0.message.tool_calls.0.tool_call.function.name")
            == agentspec_tool.name
        )
        assert (
            attrs.pop("llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments")
            == '{"x": 1, "y": 2}'
        )

        # Token counts
        assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 3
        assert attrs.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 5

    trace = get_trace()
    assert trace is None


def test_combined_agent_llm_tool_spans_are_exported(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    agentspec_agent: Agent,
    agentspec_llm_config: LlmConfig,
    agentspec_tool: Tool,
) -> None:
    """
    Combined agentic flow:
      - Open an Agent span
      - In that span, create a LLM span requesting a tool call
      - Create a Tool span that executes the tool requested by the LLM
      - Create another LLM span where the tool response is fed back into the prompt,
        and a response is simulated
      - Close the Agent span
    """

    with AgentSpecInstrumentor().instrument_context(
        skip_dep_check=True, tracer_provider=tracer_provider
    ):
        trace = get_trace()
        assert isinstance(trace, AgentSpecTrace)
        assert len(trace.span_processors) == 1

        with AgentExecutionSpan(agent=agentspec_agent) as agent_span:
            # Add agent I/O
            agent_span.add_event(
                AgentExecutionStart(agent=agentspec_agent, inputs={"question": "Compute 1+2"})
            )

            # 1) First LLM span requesting a tool call
            with LlmGenerationSpan(llm_config=agentspec_llm_config) as llm_span_1:
                req_id_1 = "req-llm-1"
                tool_call_id = "call-add-1"
                llm_span_1.add_event(
                    LlmGenerationRequest(
                        llm_config=agentspec_llm_config,
                        llm_generation_config=LlmGenerationConfig(temperature=0.2),
                        request_id=req_id_1,
                        prompt=[Message(role="user", content="Compute 1+2")],
                        tools=[agentspec_tool],
                    )
                )
                llm_span_1.add_event(
                    LlmGenerationResponse(
                        llm_config=agentspec_llm_config,
                        request_id=req_id_1,
                        content="",
                        tool_calls=[
                            ToolCall(
                                call_id=tool_call_id,
                                tool_name=agentspec_tool.name,
                                arguments='{"x": 1, "y": 2}',
                            ),
                        ],
                        input_tokens=15,
                        output_tokens=3,
                    )
                )

            # 2) Tool span executing the requested tool call
            with ToolExecutionSpan(tool=agentspec_tool) as tool_span:
                tool_span.add_event(
                    ToolExecutionRequest(
                        tool=agentspec_tool,
                        request_id=tool_call_id,
                        inputs={"x": 1, "y": 2},
                    )
                )
                tool_span.add_event(
                    ToolExecutionResponse(
                        tool=agentspec_tool,
                        request_id=tool_call_id,
                        outputs={"z": 3},
                    )
                )

            # 3) Second LLM span where the tool response is fed back into the prompt
            with LlmGenerationSpan(llm_config=agentspec_llm_config) as llm_span_2:
                req_id_2 = "req-llm-2"
                llm_span_2.add_event(
                    LlmGenerationRequest(
                        llm_config=agentspec_llm_config,
                        llm_generation_config=LlmGenerationConfig(temperature=0.0),
                        request_id=req_id_2,
                        prompt=[Message(role="user", content="Tool add returned: 3")],
                        tools=[],
                    )
                )
                llm_span_2.add_event(
                    LlmGenerationResponse(
                        llm_config=agentspec_llm_config,
                        request_id=req_id_2,
                        content="3",
                        tool_calls=[],
                        input_tokens=14,
                        output_tokens=1,
                    )
                )

            # Close Agent span with final output
            agent_span.add_event(
                AgentExecutionEnd(agent=agentspec_agent, outputs={"final_answer": "3"})
            )

        # Validate exported spans
        spans = in_memory_span_exporter.get_finished_spans()
        assert len(spans) == 4

        # Group spans by kind for stable assertions
        by_kind: Dict[str, list[Any]] = {}
        for s in spans:
            kind = str((s.attributes or {}).get(SpanAttributes.OPENINFERENCE_SPAN_KIND))
            by_kind.setdefault(kind, []).append(s)

        assert len(by_kind[OpenInferenceSpanKindValues.AGENT.value]) == 1
        assert len(by_kind[OpenInferenceSpanKindValues.TOOL.value]) == 1
        assert len(by_kind[OpenInferenceSpanKindValues.LLM.value]) == 2

        # Assert Agent span
        agent_exported = by_kind[OpenInferenceSpanKindValues.AGENT.value][0]
        agent_attrs = dict(agent_exported.attributes or {})
        assert agent_attrs.pop(SpanAttributes.AGENT_NAME) == "assistant"
        assert agent_attrs.pop(SpanAttributes.INPUT_VALUE) == '{"question": "Compute 1+2"}'
        assert agent_attrs.pop(SpanAttributes.OUTPUT_VALUE) == '{"final_answer": "3"}'

        # Identify first LLM (with tool call) and second LLM (with tool response in prompt)
        llm_spans = by_kind[OpenInferenceSpanKindValues.LLM.value]
        llm_with_tool = None
        llm_final = None
        for llm in llm_spans:
            a = dict(llm.attributes or {})
            # Presence of tool call fields indicates first LLM
            if a.get("llm.output_messages.0.message.tool_calls.0.tool_call.id"):
                llm_with_tool = a
            else:
                llm_final = a

        assert llm_with_tool is not None and llm_final is not None

        # Assert first LLM span (tool call requested)
        assert llm_with_tool.pop(SpanAttributes.LLM_MODEL_NAME) == agentspec_llm_config.model_id
        assert llm_with_tool.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS) == (
            '{"temperature": 0.2}'
        )
        assert llm_with_tool.pop("llm.input_messages.0.message.content") == "Compute 1+2"
        assert (
            llm_with_tool.pop("llm.output_messages.0.message.tool_calls.0.tool_call.id")
            == tool_call_id
        )
        assert (
            llm_with_tool.pop("llm.output_messages.0.message.tool_calls.0.tool_call.function.name")
            == agentspec_tool.name
        )
        assert (
            llm_with_tool.pop(
                "llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments"
            )
            == '{"x": 1, "y": 2}'
        )
        assert llm_with_tool.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 15
        assert llm_with_tool.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 3

        # Assert Tool span
        tool_exported = by_kind[OpenInferenceSpanKindValues.TOOL.value][0]
        tool_attrs = dict(tool_exported.attributes or {})
        assert tool_attrs.pop(SpanAttributes.TOOL_NAME) == agentspec_tool.name
        assert tool_attrs.pop(SpanAttributes.TOOL_DESCRIPTION) == agentspec_tool.description
        assert tool_attrs.pop("tool.parameters.x") == 1
        assert tool_attrs.pop("tool.parameters.y") == 2
        assert tool_attrs.pop(ToolCallAttributes.TOOL_CALL_ID) == tool_call_id
        assert tool_attrs.pop(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME) == agentspec_tool.name
        assert (
            tool_attrs.pop(ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON)
            == '{"x": 1, "y": 2}'
        )
        assert tool_attrs.pop(SpanAttributes.INPUT_VALUE) == '{"x": 1, "y": 2}'
        assert tool_attrs.pop(SpanAttributes.OUTPUT_VALUE) == '{"z": 3}'

        # Assert second LLM span (tool response sent in prompt)

        # We set temperature=0.0 explicitly
        assert llm_final.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS) == '{"temperature": 0.0}'
        assert llm_final.pop("llm.input_messages.0.message.content") == "Tool add returned: 3"
        assert llm_final.pop("llm.output_messages.0.message.content") == "3"
        assert llm_final.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 14
        assert llm_final.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 1

    # After context, trace should be cleared
    trace = get_trace()
    assert trace is None


def test_instrumentor_warns_when_no_span_processors_is_in_tracer_provider() -> None:
    provider = OtelTracerProvider()  # no span processors attached

    with pytest.warns(
        UserWarning, match="Instrumenting a TracerProvider that has no SpanProcessors attached"
    ):
        AgentSpecInstrumentor().instrument(skip_dep_check=True, tracer_provider=provider)

    trace = get_trace()
    assert isinstance(trace, AgentSpecTrace)
    AgentSpecInstrumentor().uninstrument()
    trace = get_trace()
    assert trace is None


def test_mask_sensitive_information_is_triggered(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    agentspec_agent: Agent,
) -> None:
    AgentSpecInstrumentor().instrument(
        skip_dep_check=True, mask_sensitive_information=True, tracer_provider=tracer_provider
    )

    trace = get_trace()
    assert isinstance(trace, AgentSpecTrace)
    assert len(trace.span_processors) == 1

    with AgentExecutionSpan(agent=agentspec_agent) as span:
        span.add_event(AgentExecutionStart(agent=agentspec_agent, inputs={"question": "hi"}))
        span.add_event(AgentExecutionEnd(agent=agentspec_agent, outputs={"answer": "hello"}))

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert (
        attrs.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == OpenInferenceSpanKindValues.AGENT.value
    )
    assert attrs.pop(SpanAttributes.AGENT_NAME) == "assistant"
    assert attrs.pop(SpanAttributes.INPUT_VALUE) == _PII_MASK
    assert attrs.pop(SpanAttributes.OUTPUT_VALUE) == _PII_MASK

    # Cleanup
    AgentSpecInstrumentor().uninstrument()
    trace = get_trace()
    assert trace is None


def test_instrumentor_forwards_resource_to_spans(
    in_memory_span_exporter: InMemorySpanExporter,
    agentspec_agent: Agent,
) -> None:
    provider = OtelTracerProvider(resource=OtelSdkResource(attributes={"service.name": "svc"}))
    span_processor = SimpleSpanProcessor(in_memory_span_exporter)
    provider.add_span_processor(span_processor)

    with AgentSpecInstrumentor().instrument_context(skip_dep_check=True, tracer_provider=provider):
        trace = get_trace()
        assert isinstance(trace, AgentSpecTrace)
        assert len(trace.span_processors) == 1
        with AgentExecutionSpan(agent=agentspec_agent) as span:
            span.add_event(AgentExecutionStart(agent=agentspec_agent, inputs={"question": "hi"}))
            span.add_event(AgentExecutionEnd(agent=agentspec_agent, outputs={"answer": "hello"}))

    trace = get_trace()
    assert trace is None
    for span in in_memory_span_exporter.get_finished_spans():
        attrs_res = getattr(span.resource, "attributes", {})
        assert "service.name" in attrs_res


def test_broken_exporter_and_spanprocessor_dont_disrupt_execution(
    agentspec_agent: Agent,
) -> None:
    from opentelemetry.sdk.trace.export import SpanExporter, SpanProcessor  # type: ignore

    class BrokenSpanExporter(SpanExporter):
        """Span exporter that raises exceptions in every method"""

        def export(self, spans: Any) -> Any:
            raise RuntimeError("Error during export")

        def shutdown(self) -> None:
            raise RuntimeError("Error during shutdown")

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            raise RuntimeError("Error during force_flush")

    class BrokenSpanProcessor(SpanProcessor):
        """Span processor that raises exceptions in every method"""

        def __init__(self, span_exporter: SpanExporter) -> None:
            self.span_exporter = span_exporter

        def on_start(self, span: Any, parent_context: Any = None) -> None:
            raise RuntimeError("Error during on_start")

        def on_end(self, span: Any) -> None:
            # Export on span end is what is typically done in OTel SpanProcessors
            self.span_exporter.export(span)
            raise RuntimeError("Error during on_start")

        def shutdown(self) -> None:
            # We don't raise at shutdown because the opentelemetry's TracerProvider
            # shutdown method does not catch exceptions
            return

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            raise RuntimeError("Error during force_flush")

    # We ensure that we reach the end of the test without raising any exception
    # Because they should be all swallowed to let the agent's execution go smoothly

    provider = OtelTracerProvider()
    span_processor = BrokenSpanProcessor(BrokenSpanExporter())
    provider.add_span_processor(span_processor)

    with AgentSpecInstrumentor().instrument_context(skip_dep_check=True, tracer_provider=provider):
        trace = get_trace()
        assert isinstance(trace, AgentSpecTrace)
        assert len(trace.span_processors) == 1
        with AgentExecutionSpan(agent=agentspec_agent) as span:
            span.add_event(AgentExecutionStart(agent=agentspec_agent, inputs={"question": "hi"}))
            span.add_event(AgentExecutionEnd(agent=agentspec_agent, outputs={"answer": "hello"}))

    trace = get_trace()
    assert trace is None


def test_setting_trace_suppression_stops_trace_emission(
    in_memory_span_exporter: InMemorySpanExporter,
    agentspec_agent: Agent,
) -> None:
    from opentelemetry import context as context_api

    token = context_api.attach(
        context_api.set_value(context_api._SUPPRESS_INSTRUMENTATION_KEY, True)
    )

    try:
        provider = OtelTracerProvider()
        span_processor = SimpleSpanProcessor(in_memory_span_exporter)
        provider.add_span_processor(span_processor)

        with AgentSpecInstrumentor().instrument_context(
            skip_dep_check=True, tracer_provider=provider
        ):
            trace = get_trace()
            assert isinstance(trace, AgentSpecTrace)
            assert len(trace.span_processors) == 1
            with AgentExecutionSpan(agent=agentspec_agent) as span:
                span.add_event(
                    AgentExecutionStart(agent=agentspec_agent, inputs={"question": "hi"})
                )
                span.add_event(
                    AgentExecutionEnd(agent=agentspec_agent, outputs={"answer": "hello"})
                )

        assert len(in_memory_span_exporter.get_finished_spans()) == 0
        trace = get_trace()
        assert trace is None
    finally:
        context_api.detach(token)
