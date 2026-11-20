"""End-to-end tests for the `tool_span_kinds` instrument() option."""

from __future__ import annotations

import json
from typing import Any

import pytest
from agents import Agent, Runner, function_tool
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor


@function_tool
def search_documents(query: str) -> list[dict[str, Any]]:
    """Search the knowledge base."""
    return [
        {"id": "doc-1", "content": f"About {query}", "score": 0.9, "source": "kb"},
        {"id": "doc-2", "content": "Something else", "score": 0.4},
    ]


class _FakeModel(Model):
    """Calls search_documents on the first turn, then answers."""

    def __init__(self) -> None:
        self.calls = 0

    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        output: list[Any]
        if self.calls == 1:
            output = [
                ResponseFunctionToolCall(
                    type="function_call",
                    call_id="call-1",
                    name="search_documents",
                    arguments='{"query":"France"}',
                )
            ]
        else:
            output = [
                ResponseOutputMessage(
                    id="m1",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[ResponseOutputText(type="output_text", text="Done.", annotations=[])],
                )
            ]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def exporter() -> Any:
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    OpenAIAgentsInstrumentor().instrument(
        tracer_provider=provider,
        tool_span_kinds={"search_documents": "RETRIEVER"},
    )
    yield exporter
    OpenAIAgentsInstrumentor().uninstrument()


async def test_function_tool_mapped_to_retriever_end_to_end(
    exporter: InMemorySpanExporter,
) -> None:
    agent = Agent(name="SearchAgent", model=_FakeModel(), tools=[search_documents])

    result = await Runner.run(agent, "Tell me about France")
    assert result.final_output == "Done."

    spans = [s for s in exporter.get_finished_spans() if s.name == "search_documents"]
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs["openinference.span.kind"] == "RETRIEVER"
    assert attrs["tool.name"] == "search_documents"
    assert json.loads(str(attrs["input.value"])) == {"query": "France"}
    assert attrs["retrieval.documents.0.document.id"] == "doc-1"
    assert attrs["retrieval.documents.0.document.content"] == "About France"
    assert attrs["retrieval.documents.0.document.score"] == 0.9
    assert json.loads(str(attrs["retrieval.documents.0.document.metadata"])) == {"source": "kb"}
    assert attrs["retrieval.documents.1.document.id"] == "doc-2"


def test_instrument_rejects_unsupported_tool_span_kind() -> None:
    instrumentor = OpenAIAgentsInstrumentor()
    with pytest.raises(ValueError, match="tool_span_kinds"):
        instrumentor.instrument(tool_span_kinds={"embed_text": "EMBEDDING"})
    assert not instrumentor.is_instrumented_by_opentelemetry
