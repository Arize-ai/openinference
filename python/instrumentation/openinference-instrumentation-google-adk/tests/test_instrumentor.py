from typing import Iterator
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace as trace_api
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points

from openinference.instrumentation.google_adk import GoogleADKInstrumentor


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(
    in_memory_span_exporter: InMemorySpanExporter,
) -> trace_api.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider()
    span_processor = SimpleSpanProcessor(span_exporter=in_memory_span_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    return tracer_provider


@pytest.fixture
def instrument(
    tracer_provider: trace_api.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> Iterator[None]:
    GoogleADKInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    GoogleADKInstrumentor().uninstrument()


def test_entrypoint_for_opentelemetry_instrument() -> None:
    (instrumentor_entrypoint,) = entry_points(group="opentelemetry_instrumentor", name="google_adk")
    instrumentor = instrumentor_entrypoint.load()()
    assert isinstance(instrumentor, GoogleADKInstrumentor)


def test_google_adk_instrumentor(instrument, in_memory_span_exporter: InMemorySpanExporter):
    from google.adk.agents.llm_agent import LlmAgent

    agent = LlmAgent(name="agent")
    context = MagicMock()
    context.agent_name = "agent"

    llm_request = MagicMock()
    llm_request.model = "model"

    llm_response = MagicMock()
    llm_response.content.model_dump.return_value = {"response": "yes"}

    tool = MagicMock()
    tool.name = "tool"

    agent.before_agent_callback(context)

    agent.before_model_callback(context, llm_request)
    agent.after_model_callback(context, llm_response)

    agent.before_tool_callback(tool, {}, context)
    agent.after_tool_callback(tool, {}, context, "tool response")

    agent.after_agent_callback(context)

    assert len(in_memory_span_exporter._finished_spans) == 3
    assert in_memory_span_exporter._finished_spans[0].name == "agent-LLM"
    assert in_memory_span_exporter._finished_spans[1].name == "agent-TOOL"
    assert in_memory_span_exporter._finished_spans[2].name == "agent-AGENT"

    assert in_memory_span_exporter._finished_spans[0].attributes["llm.model_name"] == "model"
    assert in_memory_span_exporter._finished_spans[1].attributes["tool.name"] == "tool"

    parent_id = in_memory_span_exporter._finished_spans[2].context.span_id
    assert in_memory_span_exporter._finished_spans[0].parent.span_id == parent_id
    assert in_memory_span_exporter._finished_spans[1].parent.span_id == parent_id
