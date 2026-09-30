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
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Hi there."
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == "text/plain"
    assert attributes.pop(SpanAttributes.LLM_FINISH_REASON) == "end_turn"
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
