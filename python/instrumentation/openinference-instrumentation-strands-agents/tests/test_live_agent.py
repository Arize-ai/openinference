"""Run real Strands agents against OpenAI and check the spans the processor exports.

The OpenAI calls replay from the cassettes in `tests/cassettes/test_live_agent/`, so no key is
needed. To record them again, delete the cassettes and run with a real key:

    OPENAI_API_KEY=... pytest tests/test_live_agent.py --record-mode=once
"""

import inspect
import json
import struct
import zlib
from typing import Any, Dict, List

import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.telemetry import tracer as strands_tracer

from openinference.instrumentation.strands_agents import StrandsAgentsToOpenInferenceProcessor
from openinference.semconv.trace import SpanAttributes

MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a weather bot. Answer in one short sentence."
QUESTION = "What is the weather in Paris?"
WEATHER = "The weather in Paris is sunny and 72F."
OPT_IN = {"legacy": "", "latest": "gen_ai_latest_experimental"}


def llm_output(content: str, finish_reason: str, model: str = "gpt-4") -> Dict[str, Any]:
    """The `output.value` of an LLM span: the reply as a single OpenAI-style choice."""
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "index": 0,
                "message": {"content": content, "role": "assistant"},
            }
        ],
        "model": model,
        "usage": {"completion_tokens": None, "prompt_tokens": None, "total_tokens": None},
    }


# Raw Strands attributes that hold prompt or message content.
CONTENT_KEYS = {
    "gen_ai.system_instructions",
    "gen_ai.input.messages",
    "gen_ai.output.messages",
    "system_prompt",
}
OPENINFERENCE_PREFIXES = (
    SpanAttributes.LLM_INPUT_MESSAGES,
    SpanAttributes.LLM_OUTPUT_MESSAGES,
    "input.",
    "output.",
    SpanAttributes.LLM_FINISH_REASON,
)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city
    """
    return f"The weather in {city} is sunny and 72F."


def use_strands_tracer(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    mode: str,
) -> None:
    """Send the spans of agents created afterwards through the processor into `exporter`."""
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", OPT_IN[mode])
    tracer = strands_tracer.Tracer()
    if mode == "latest" and not getattr(tracer, "use_latest_genai_conventions", False):
        pytest.skip("this Strands version has no latest GenAI conventions")
    tracer.tracer = tracer_provider.get_tracer("strands-test")
    # Strands looks the tracer up through this module-level singleton.
    monkeypatch.setattr(strands_tracer, "_tracer_instance", tracer)
    tracer_provider.add_span_processor(StrandsAgentsToOpenInferenceProcessor())
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))


def system_prompt_on_model_spans() -> bool:
    # Strands added the system prompt to model spans in 1.34.
    parameters = inspect.signature(strands_tracer.Tracer.start_model_invoke_span).parameters
    return "system_prompt" in parameters


def spans_of_kind(exporter: InMemorySpanExporter, kind: str) -> List[ReadableSpan]:
    return [
        span
        for span in exporter.get_finished_spans()
        if (span.attributes or {}).get(SpanAttributes.OPENINFERENCE_SPAN_KIND) == kind
    ]


def openinference_attributes(span: ReadableSpan) -> Dict[str, Any]:
    attributes = dict(span.attributes or {})
    # Messages are only flattened, and their content is not copied into metadata.
    assert SpanAttributes.LLM_INPUT_MESSAGES not in attributes
    assert SpanAttributes.LLM_OUTPUT_MESSAGES not in attributes
    metadata = json.loads(str(attributes.get(SpanAttributes.METADATA, "{}")))
    assert not CONTENT_KEYS & set(metadata)
    return {k: v for k, v in attributes.items() if k.startswith(OPENINFERENCE_PREFIXES)}


def pop_text(attributes: Dict[str, Any], key: str) -> str:
    value = attributes.pop(key)
    assert isinstance(value, str) and value
    return value


def tiny_png() -> bytes:
    """An 8x8 red PNG."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * 8 for _ in range(8))
    header = struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


@pytest.mark.vcr
@pytest.mark.parametrize("mode", list(OPT_IN))
def test_agent_with_system_prompt_and_tool(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    use_strands_tracer(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    agent = Agent(
        model=OpenAIModel(model_id=MODEL, params={"temperature": 0}),
        tools=[get_weather],
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,
    )
    agent(QUESTION)

    system = (
        [{"role": "system", "content": SYSTEM_PROMPT}] if system_prompt_on_model_spans() else []
    )
    prefix_in = SpanAttributes.LLM_INPUT_MESSAGES
    prefix_out = SpanAttributes.LLM_OUTPUT_MESSAGES

    call_span, answer_span = spans_of_kind(in_memory_span_exporter, "LLM")

    # First model call: the system prompt and the question, answered with a tool call.
    attributes = openinference_attributes(call_span)
    for i, message in enumerate([*system, {"role": "user", "content": QUESTION}]):
        assert attributes.pop(f"{prefix_in}.{i}.message.role") == message["role"]
        assert attributes.pop(f"{prefix_in}.{i}.message.content") == message["content"]
    call = f"{prefix_out}.0.message.tool_calls.0.tool_call"
    call_id = pop_text(attributes, f"{call}.id")
    assert attributes.pop(f"{call}.function.name") == "get_weather"
    assert json.loads(attributes.pop(f"{call}.function.arguments")) == {"city": "Paris"}
    assert attributes.pop(f"{prefix_out}.0.message.role") == "assistant"
    assert attributes.pop(f"{prefix_out}.0.message.finish_reason") == "tool_use"
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == QUESTION
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
    assert json.loads(attributes.pop(SpanAttributes.OUTPUT_VALUE)) == llm_output(
        "", "tool_use", model=MODEL
    )
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert not attributes

    # Second model call: the whole tool loop goes in, the final answer comes out.
    attributes = openinference_attributes(answer_span)
    i = 0
    for message in system:
        assert attributes.pop(f"{prefix_in}.{i}.message.role") == message["role"]
        assert attributes.pop(f"{prefix_in}.{i}.message.content") == message["content"]
        i += 1
    assert attributes.pop(f"{prefix_in}.{i}.message.role") == "user"
    assert attributes.pop(f"{prefix_in}.{i}.message.content") == QUESTION
    i += 1
    assert attributes.pop(f"{prefix_in}.{i}.message.role") == "assistant"
    history_call = f"{prefix_in}.{i}.message.tool_calls.0.tool_call"
    assert attributes.pop(f"{history_call}.id") == call_id
    assert attributes.pop(f"{history_call}.function.name") == "get_weather"
    assert json.loads(attributes.pop(f"{history_call}.function.arguments")) == {"city": "Paris"}
    i += 1
    assert attributes.pop(f"{prefix_in}.{i}.message.role") == "tool"
    assert attributes.pop(f"{prefix_in}.{i}.message.content") == WEATHER
    assert attributes.pop(f"{prefix_in}.{i}.message.tool_call_id") == call_id
    assert attributes.pop(f"{prefix_in}.{i}.message.name") == "get_weather"
    assert attributes.pop(f"{prefix_out}.0.message.role") == "assistant"
    answer = pop_text(attributes, f"{prefix_out}.0.message.content")
    assert attributes.pop(f"{prefix_out}.0.message.finish_reason") == "end_turn"
    sent = json.loads(attributes.pop(SpanAttributes.INPUT_VALUE))
    assert [m["message.role"] for m in sent["messages"]] == [
        *(m["role"] for m in system),
        "user",
        "assistant",
        "tool",
    ]
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert json.loads(attributes.pop(SpanAttributes.OUTPUT_VALUE)) == llm_output(
        answer, "end_turn", model=MODEL
    )
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert not attributes

    # The agent span always carries the system prompt, on every Strands version.
    (agent_span,) = spans_of_kind(in_memory_span_exporter, "AGENT")
    attributes = openinference_attributes(agent_span)
    assert attributes.pop(f"{prefix_in}.0.message.role") == "system"
    assert attributes.pop(f"{prefix_in}.0.message.content") == SYSTEM_PROMPT
    assert attributes.pop(f"{prefix_in}.1.message.role") == "user"
    assert attributes.pop(f"{prefix_in}.1.message.content") == QUESTION
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == QUESTION
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(f"{prefix_out}.0.message.role") == "assistant"
    assert pop_text(attributes, f"{prefix_out}.0.message.content").strip() == answer.strip()
    assert pop_text(attributes, SpanAttributes.OUTPUT_VALUE).strip() == answer.strip()
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
    attributes.pop(f"{prefix_out}.0.message.finish_reason", None)
    assert not attributes

    (tool_span,) = spans_of_kind(in_memory_span_exporter, "TOOL")
    tool_attributes = dict(tool_span.attributes or {})
    assert tool_attributes[SpanAttributes.TOOL_NAME] == "get_weather"
    assert json.loads(str(tool_attributes[SpanAttributes.INPUT_VALUE])) == {"city": "Paris"}
    assert tool_attributes[SpanAttributes.OUTPUT_VALUE] == WEATHER


@pytest.mark.vcr
@pytest.mark.parametrize("mode", list(OPT_IN))
def test_agent_with_inline_image(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    use_strands_tracer(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    agent = Agent(
        model=OpenAIModel(model_id=MODEL, params={"temperature": 0}),
        callback_handler=None,
    )
    prompt: Any = [
        {"text": "What color is this image? Answer in one word."},
        {"image": {"format": "png", "source": {"bytes": tiny_png()}}},
    ]
    agent(prompt)

    (llm_span,) = spans_of_kind(in_memory_span_exporter, "LLM")
    attributes = openinference_attributes(llm_span)
    contents = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.contents"
    # Strands replaces the image bytes before they reach the span, so the block is kept as JSON.
    placeholder = json.dumps({"image": {"format": "png", "source": {"bytes": "<replaced>"}}})
    assert attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role") == "user"
    assert attributes.pop(f"{contents}.0.message_content.type") == "text"
    assert attributes.pop(f"{contents}.0.message_content.text") == prompt[0]["text"]
    assert attributes.pop(f"{contents}.1.message_content.type") == "text"
    assert attributes.pop(f"{contents}.1.message_content.text") == placeholder
    out = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.message"
    assert attributes.pop(f"{out}.role") == "assistant"
    answer = pop_text(attributes, f"{out}.content")
    assert "red" in answer.lower()
    assert attributes.pop(f"{out}.finish_reason") == "end_turn"
    sent = json.loads(attributes.pop(SpanAttributes.INPUT_VALUE))
    assert sent["messages"][0]["message.contents"][1]["message_content.text"] == placeholder
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert json.loads(attributes.pop(SpanAttributes.OUTPUT_VALUE)) == llm_output(
        answer, "end_turn", model=MODEL
    )
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert not attributes
