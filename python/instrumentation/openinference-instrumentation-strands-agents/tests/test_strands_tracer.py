"""Run spans from the installed Strands tracer through the processor.

Nothing here calls a model. The spans come from `strands.telemetry.tracer.Tracer` itself, so a
change in what Strands records shows up when the `-latest` target installs a newer release.
"""

import inspect
import json
from typing import Any, Dict, List, Optional

import pytest
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from strands.telemetry.tracer import Tracer

from openinference.instrumentation import TraceConfig
from openinference.instrumentation.config import REDACTED_VALUE
from openinference.instrumentation.strands_agents import StrandsAgentsToOpenInferenceProcessor
from openinference.semconv.trace import SpanAttributes

OPT_IN = {
    "legacy": "",
    "latest": "gen_ai_latest_experimental",
    "attributes_only": "gen_ai_latest_experimental,gen_ai_span_attributes_only",
}
MODES = list(OPT_IN)

SYSTEM_PROMPT = "You are concise."
USER_MESSAGES: Any = [{"role": "user", "content": [{"text": "Hello"}]}]
ASSISTANT_MESSAGE: Any = {"role": "assistant", "content": [{"text": "Hi there."}]}
USAGE: Any = {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8}
METRICS: Any = {"latencyMs": 1}


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


class Harness:
    def __init__(self, tracer: Tracer, exporter: InMemorySpanExporter) -> None:
        self.tracer = tracer
        self.exporter = exporter

    @property
    def takes_system_prompt_on_model_span(self) -> bool:
        # Strands 1.19 only records the system prompt on the agent span.
        parameters = inspect.signature(self.tracer.start_model_invoke_span).parameters
        return "system_prompt" in parameters

    def only_span(self) -> ReadableSpan:
        (span,) = self.exporter.get_finished_spans()
        return span


def make_harness(
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    exporter: InMemorySpanExporter,
    mode: str,
    config: Optional[TraceConfig] = None,
) -> Harness:
    monkeypatch.setenv("OTEL_SEMCONV_STABILITY_OPT_IN", OPT_IN[mode])
    tracer = Tracer()
    if mode != "legacy" and not getattr(tracer, "use_latest_genai_conventions", False):
        pytest.skip("this Strands version has no latest GenAI conventions")
    if mode == "attributes_only" and not getattr(tracer, "_span_attributes_only", False):
        pytest.skip("this Strands version cannot record messages as span attributes")
    tracer.tracer = tracer_provider.get_tracer("strands-test")
    # The processor must come before the exporter: it rewrites the span in place.
    tracer_provider.add_span_processor(StrandsAgentsToOpenInferenceProcessor(config=config))
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    return Harness(tracer, exporter)


def openinference_attributes(span: ReadableSpan) -> Dict[str, Any]:
    attributes = dict(span.attributes or {})
    # Prompt and message content must not be copied into metadata.
    metadata = json.loads(str(attributes.get(SpanAttributes.METADATA, "{}")))
    assert not CONTENT_KEYS & set(metadata)
    return {k: v for k, v in attributes.items() if k.startswith(OPENINFERENCE_PREFIXES)}


def pop_messages(attributes: Dict[str, Any], prefix: str, messages: List[Dict[str, Any]]) -> None:
    for i, message in enumerate(messages):
        for key, value in message.items():
            assert attributes.pop(f"{prefix}.{i}.message.{key}") == value


def run_model_span(harness: Harness) -> None:
    kwargs: Dict[str, Any] = {}
    if harness.takes_system_prompt_on_model_span:
        kwargs["system_prompt"] = SYSTEM_PROMPT
    span = harness.tracer.start_model_invoke_span(USER_MESSAGES, model_id="gpt-4", **kwargs)
    harness.tracer.end_model_invoke_span(span, ASSISTANT_MESSAGE, USAGE, METRICS, "end_turn")


@pytest.mark.parametrize("mode", MODES)
def test_model_span(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    run_model_span(harness)

    span = harness.only_span()
    assert span.attributes
    assert span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "LLM"
    attributes = openinference_attributes(span)
    expected_input = [{"role": "user", "content": "Hello"}]
    if harness.takes_system_prompt_on_model_span:
        expected_input.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    pop_messages(attributes, SpanAttributes.LLM_INPUT_MESSAGES, expected_input)
    pop_messages(
        attributes,
        SpanAttributes.LLM_OUTPUT_MESSAGES,
        [{"role": "assistant", "content": "Hi there.", "finish_reason": "end_turn"}],
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hello"
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
    assert json.loads(attributes.pop(SpanAttributes.OUTPUT_VALUE)) == llm_output(
        "Hi there.", "end_turn"
    )
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert not attributes


@pytest.mark.parametrize("mode", MODES)
def test_agent_span_has_the_system_prompt(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    span = harness.tracer.start_agent_span(
        USER_MESSAGES, "A", model_id="gpt-4", system_prompt=SYSTEM_PROMPT
    )
    span.end()

    exported = harness.only_span()
    assert exported.attributes
    assert exported.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "AGENT"
    attributes = openinference_attributes(exported)
    pop_messages(
        attributes,
        SpanAttributes.LLM_INPUT_MESSAGES,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == "Hello"
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "text/plain"
    assert not attributes


@pytest.mark.parametrize("mode", MODES)
def test_hide_inputs_removes_the_system_prompt_everywhere(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(
        monkeypatch,
        tracer_provider,
        in_memory_span_exporter,
        mode,
        config=TraceConfig(hide_inputs=True),
    )
    span = harness.tracer.start_agent_span(
        USER_MESSAGES, "A", model_id="gpt-4", system_prompt=SYSTEM_PROMPT
    )
    span.end()

    exported = harness.only_span()
    attributes = dict(exported.attributes or {})
    # Neither the raw Strands attributes nor the converted ones keep the prompt or the user text.
    assert not CONTENT_KEYS & set(attributes)
    assert not [k for k in attributes if k.startswith(SpanAttributes.LLM_INPUT_MESSAGES)]
    assert attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert SYSTEM_PROMPT not in json.dumps(attributes, default=str)
    assert "Hello" not in json.dumps(attributes, default=str)


@pytest.mark.parametrize("mode", MODES)
def test_image_only_message_is_not_dropped(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    image: Any = {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}
    messages: Any = [{"role": "user", "content": [image]}]
    span = harness.tracer.start_model_invoke_span(messages, model_id="gpt-4")
    harness.tracer.end_model_invoke_span(span, ASSISTANT_MESSAGE, USAGE, METRICS, "end_turn")

    attributes = openinference_attributes(harness.only_span())
    # Strands replaces the raw bytes before they reach the span, so the block is kept as JSON.
    placeholder = json.dumps({"image": {"format": "png", "source": {"bytes": "<replaced>"}}})
    prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message"
    assert attributes.pop(f"{prefix}.role") == "user"
    assert attributes.pop(f"{prefix}.contents.0.message_content.type") == "text"
    assert attributes.pop(f"{prefix}.contents.0.message_content.text") == placeholder
    pop_messages(
        attributes,
        SpanAttributes.LLM_OUTPUT_MESSAGES,
        [{"role": "assistant", "content": "Hi there.", "finish_reason": "end_turn"}],
    )
    assert json.loads(attributes.pop(SpanAttributes.INPUT_VALUE)) == {
        "messages": [
            {
                "message.role": "user",
                "message.contents": [
                    {"message_content.type": "text", "message_content.text": placeholder}
                ],
            }
        ],
        "model": "gpt-4",
    }
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert json.loads(attributes.pop(SpanAttributes.OUTPUT_VALUE)) == llm_output(
        "Hi there.", "end_turn"
    )
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "application/json"
    assert not attributes


@pytest.mark.parametrize("mode", MODES)
def test_system_prompt_blocks_keep_line_breaks(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    if (
        "system_prompt_content"
        not in inspect.signature(harness.tracer.start_model_invoke_span).parameters
    ):
        pytest.skip("this Strands version has no structured system prompt on model spans")
    blocks: Any = [{"text": "Be brief."}, {"text": "No emoji."}]
    span = harness.tracer.start_model_invoke_span(
        USER_MESSAGES,
        model_id="gpt-4",
        system_prompt="Be brief.\nNo emoji.",
        system_prompt_content=blocks,
    )
    harness.tracer.end_model_invoke_span(span, ASSISTANT_MESSAGE, USAGE, METRICS, "end_turn")

    attributes = openinference_attributes(harness.only_span())
    prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message"
    assert attributes.pop(f"{prefix}.role") == "system"
    assert attributes.pop(f"{prefix}.content") == "Be brief.\nNo emoji."


@pytest.mark.parametrize("kind", ["image", "video", "audio"])
@pytest.mark.parametrize("mode", MODES)
def test_s3_media_becomes_a_url(
    mode: str,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    location = {"type": "s3", "uri": "s3://bucket/cat.png"}
    messages: Any = [
        {
            "role": "user",
            "content": [{kind: {"format": "png", "source": {"location": location}}}],
        }
    ]
    span = harness.tracer.start_model_invoke_span(messages, model_id="gpt-4")
    harness.tracer.end_model_invoke_span(span, ASSISTANT_MESSAGE, USAGE, METRICS, "end_turn")

    attributes = openinference_attributes(harness.only_span())
    prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message"
    assert attributes.pop(f"{prefix}.role") == "user"
    assert attributes.pop(f"{prefix}.contents.0.message_content.type") == kind
    assert (
        attributes.pop(f"{prefix}.contents.0.message_content.{kind}.{kind}.url")
        == "s3://bucket/cat.png"
    )


@pytest.mark.parametrize("mode", MODES)
def test_tool_span_input_and_output(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    tool: Any = {"toolUseId": "call_1", "name": "get_weather", "input": {"city": "Paris"}}
    result: Any = {"toolUseId": "call_1", "status": "success", "content": [{"text": "Sunny"}]}
    span = harness.tracer.start_tool_call_span(tool)
    harness.tracer.end_tool_call_span(span, result)

    attributes = dict(harness.only_span().attributes or {})
    assert attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND] == "TOOL"
    assert attributes[SpanAttributes.TOOL_NAME] == "get_weather"
    assert json.loads(str(attributes[SpanAttributes.TOOL_PARAMETERS])) == {"city": "Paris"}
    assert json.loads(str(attributes[SpanAttributes.INPUT_VALUE])) == {"city": "Paris"}
    assert attributes[SpanAttributes.INPUT_MIME_TYPE] == "application/json"
    assert attributes[SpanAttributes.OUTPUT_VALUE] == "Sunny"
    assert attributes[SpanAttributes.OUTPUT_MIME_TYPE] == "text/plain"


@pytest.mark.parametrize(
    "usage",
    [
        # The provider already counts the cache in inputTokens.
        {"inputTokens": 8, "outputTokens": 2, "totalTokens": 10, "cacheReadInputTokens": 3},
        # The provider reports the cache on top of inputTokens.
        {"inputTokens": 5, "outputTokens": 2, "totalTokens": 10, "cacheReadInputTokens": 3},
    ],
)
@pytest.mark.parametrize("mode", MODES)
def test_cached_tokens_are_counted_once(
    mode: str,
    usage: Any,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    span = harness.tracer.start_model_invoke_span(USER_MESSAGES, model_id="gpt-4")
    harness.tracer.end_model_invoke_span(span, ASSISTANT_MESSAGE, usage, METRICS, "end_turn")

    attributes = dict(harness.only_span().attributes or {})
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 8
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 2
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 10
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ] == 3


@pytest.mark.parametrize("mode", MODES)
def test_zero_argument_tool_span_keeps_its_input(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    tracer_provider: trace_sdk.TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    harness = make_harness(monkeypatch, tracer_provider, in_memory_span_exporter, mode)
    tool: Any = {"toolUseId": "call_1", "name": "get_time", "input": {}}
    result: Any = {"toolUseId": "call_1", "status": "success", "content": [{"text": "12:00"}]}
    span = harness.tracer.start_tool_call_span(tool)
    harness.tracer.end_tool_call_span(span, result)

    attributes = dict(harness.only_span().attributes or {})
    assert attributes[SpanAttributes.TOOL_PARAMETERS] == "{}"
    assert attributes[SpanAttributes.INPUT_VALUE] == "{}"
    assert attributes[SpanAttributes.INPUT_MIME_TYPE] == "application/json"
    call = f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.tool_calls.0.tool_call"
    assert attributes[f"{call}.function.arguments"] == "{}"
    assert attributes[SpanAttributes.OUTPUT_VALUE] == "12:00"
