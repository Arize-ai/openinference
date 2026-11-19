"""Tests for the tools offered to the model on Chat Completions generation spans.

openai-agents 0.25.0 and later put these tools on ``GenerationSpanData.tools``. The tests
run the real ``OpenAIChatCompletionsModel`` against a mocked HTTP endpoint and check that
the recorded tools are the ones actually sent in the request.
"""

from __future__ import annotations

import json
import types
from collections.abc import Iterator
from typing import Any

import openai._base_client
import pytest
from agents import Agent, OpenAIChatCompletionsModel, RunConfig, Runner, function_tool
from agents.tracing.span_data import GenerationSpanData
from openai import AsyncOpenAI
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import suppress_tracing
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor

_BASE_URL = "http://chat-completions.test/v1"
_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "4"},
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
}
_CHUNKS = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": "4"}, "finish_reason": None}
        ],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
    },
]
_SDK_RECORDS_TOOLS = "tools" in GenerationSpanData.__slots__

# openai 2.x sends requests with httpx and openai 3.x with its fork httpx2, so the mock
# transport has to come from whichever one the installed client uses.
_http: Any = next(
    module
    for module in vars(openai._base_client).values()
    if isinstance(module, types.ModuleType) and module.__name__ in ("httpx", "httpx2")
)


@function_tool
def add(a: int, b: int) -> int:
    """Adds two numbers together."""
    return a + b


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = trace_sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    OpenAIAgentsInstrumentor().instrument(tracer_provider=provider)
    yield exporter
    OpenAIAgentsInstrumentor().uninstrument()


@pytest.fixture
def chat_completions() -> list[dict[str, Any]]:
    """The JSON bodies of the Chat Completions requests the model sends."""
    return []


def _client(requests: list[dict[str, Any]]) -> AsyncOpenAI:
    def respond(request: Any) -> Any:
        body = json.loads(request.content)
        requests.append(body)
        if body.get("stream"):
            events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in _CHUNKS)
            return _http.Response(
                200,
                content=events + "data: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return _http.Response(200, json=_COMPLETION)

    return AsyncOpenAI(
        base_url=_BASE_URL,
        api_key="sk-test",
        http_client=_http.AsyncClient(transport=_http.MockTransport(respond)),
    )


def _agent(requests: list[dict[str, Any]], **kwargs: Any) -> Agent[Any]:
    model = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=_client(requests))
    billing = Agent(name="Billing", instructions="Handle billing.", model=model)
    return Agent(
        name="Calculator",
        instructions="Use the add tool.",
        model=model,
        tools=kwargs.pop("tools", [add]),
        handoffs=kwargs.pop("handoffs", [billing]),
        **kwargs,
    )


async def _run(agent: Agent[Any], streamed: bool, run_config: Any = None) -> None:
    if streamed:
        result = Runner.run_streamed(agent, "What is 2 + 2?", run_config=run_config)
        async for _ in result.stream_events():
            pass
    else:
        await Runner.run(agent, "What is 2 + 2?", run_config=run_config)


def _generation_attributes(exporter: InMemorySpanExporter) -> dict[str, Any]:
    spans = [s for s in exporter.get_finished_spans() if s.name == "generation"]
    assert len(spans) == 1, f"expected one generation span, got {len(spans)}"
    return dict(spans[0].attributes or {})


def _recorded_tools(attributes: dict[str, Any]) -> list[Any]:
    tools = []
    i = 0
    while (schema := attributes.get(f"llm.tools.{i}.tool.json_schema")) is not None:
        tools.append(json.loads(schema))
        i += 1
    assert not [k for k in attributes if k.startswith(f"llm.tools.{i}.")]
    return tools


@pytest.mark.skipif("not _SDK_RECORDS_TOOLS")
@pytest.mark.parametrize("streamed", [False, True], ids=["get_response", "stream_response"])
async def test_generation_span_records_the_tools_sent_to_the_model(
    exporter: InMemorySpanExporter,
    chat_completions: list[dict[str, Any]],
    streamed: bool,
) -> None:
    await _run(_agent(chat_completions), streamed)

    sent = chat_completions[-1]["tools"]
    assert [t["function"]["name"] for t in sent] == ["add", "transfer_to_billing"]
    assert _recorded_tools(_generation_attributes(exporter)) == sent


@pytest.mark.parametrize("streamed", [False, True], ids=["get_response", "stream_response"])
async def test_no_tools_recorded_when_run_excludes_sensitive_data(
    exporter: InMemorySpanExporter,
    chat_completions: list[dict[str, Any]],
    streamed: bool,
) -> None:
    await _run(_agent(chat_completions), streamed, RunConfig(trace_include_sensitive_data=False))

    assert "tools" in chat_completions[-1]
    assert _recorded_tools(_generation_attributes(exporter)) == []


@pytest.mark.parametrize("streamed", [False, True], ids=["get_response", "stream_response"])
async def test_agent_without_tools_records_none(
    exporter: InMemorySpanExporter,
    chat_completions: list[dict[str, Any]],
    streamed: bool,
) -> None:
    await _run(_agent(chat_completions, tools=[], handoffs=[]), streamed)

    assert _recorded_tools(_generation_attributes(exporter)) == []


async def test_no_spans_when_tracing_suppressed(
    exporter: InMemorySpanExporter,
    chat_completions: list[dict[str, Any]],
) -> None:
    with suppress_tracing():
        await _run(_agent(chat_completions), streamed=False)

    assert chat_completions
    assert not exporter.get_finished_spans()
