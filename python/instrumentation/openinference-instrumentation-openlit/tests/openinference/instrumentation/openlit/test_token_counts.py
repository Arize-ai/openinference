from typing import Any, Dict, Iterator, Mapping, cast

import httpx
import openlit  # type: ignore[import-untyped]
import pytest
from anthropic import Anthropic
from openai import OpenAI
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.openlit import OpenInferenceSpanProcessor
from openinference.semconv.trace import (
    OpenInferenceLLMSystemValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)

PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
CACHE_WRITE = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
REASONING = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION_DETAILS_REASONING
TOKEN_COUNTS = {
    value for name, value in vars(SpanAttributes).items() if name.startswith("LLM_TOKEN_COUNT_")
}


@pytest.fixture(scope="module")
def in_memory_span_exporter(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    # openlit's instrumentors trace through the global tracer provider.
    trace_api.set_tracer_provider(tracer_provider)
    # A local price list keeps openlit.init from downloading one.
    pricing_json = tmp_path_factory.mktemp("openlit") / "pricing.json"
    pricing_json.write_text("{}")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("OTEL_TRACES_EXPORTER", "none")
        openlit.init(
            disable_metrics=True,
            disable_events=True,
            disable_batch=True,
            pricing_json=str(pricing_json),
        )
    yield exporter


def _token_counts(attributes: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in attributes.items() if key in TOKEN_COUNTS}


def _call(api: str, body: Dict[str, Any]) -> None:
    http_client = httpx.Client(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    )
    if api == "anthropic":
        Anthropic(api_key="sk-test", http_client=http_client).messages.create(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": "hello"}],
        )
    elif api == "openai-responses":
        OpenAI(api_key="sk-test", http_client=http_client).responses.create(
            model="o4-mini", input="hello"
        )
    else:
        OpenAI(api_key="sk-test", http_client=http_client).chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hello"}]
        )


def _chat_completion() -> Dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
            }
        ],
        "usage": {
            "prompt_tokens": 1200,
            "completion_tokens": 300,
            "total_tokens": 1500,
            "prompt_tokens_details": {"cached_tokens": 1000},
        },
    }


def _response() -> Dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "o4-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 1200,
            "output_tokens": 300,
            "total_tokens": 1500,
            "input_tokens_details": {"cached_tokens": 1000},
            "output_tokens_details": {"reasoning_tokens": 120},
        },
    }


def _message(**cache_usage: int) -> Dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 50, "output_tokens": 300, **cache_usage},
    }


@pytest.mark.parametrize(
    "api, body, system, expected",
    [
        pytest.param(
            "openai-chat",
            _chat_completion(),
            OpenInferenceLLMSystemValues.OPENAI.value,
            {PROMPT: 1200, COMPLETION: 300, CACHE_READ: 1000},
            id="openai-chat-cache-read",
        ),
        pytest.param(
            "openai-responses",
            _response(),
            OpenInferenceLLMSystemValues.OPENAI.value,
            {PROMPT: 1200, COMPLETION: 300, CACHE_READ: 1000, REASONING: 120},
            id="openai-responses-reasoning",
        ),
        pytest.param(
            "anthropic",
            _message(cache_read_input_tokens=1000, cache_creation_input_tokens=150),
            OpenInferenceLLMSystemValues.ANTHROPIC.value,
            {PROMPT: 50, COMPLETION: 300, CACHE_READ: 1000, CACHE_WRITE: 150},
            id="anthropic-cache-read-and-write",
        ),
        pytest.param(
            "anthropic",
            _message(),
            OpenInferenceLLMSystemValues.ANTHROPIC.value,
            {PROMPT: 50, COMPLETION: 300},
            id="anthropic-no-cache",
        ),
    ],
)
def test_openlit_llm_span_token_counts(
    in_memory_span_exporter: InMemorySpanExporter,
    api: str,
    body: Dict[str, Any],
    system: str,
    expected: Dict[str, int],
) -> None:
    in_memory_span_exporter.clear()
    _call(api, body)
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(cast(Mapping[str, Any], spans[0].attributes))
    assert "gen_ai.system" not in attributes
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.get(SpanAttributes.LLM_SYSTEM) == system
    assert _token_counts(attributes) == expected


@pytest.mark.parametrize("system_key", ["gen_ai.system", "gen_ai.provider.name"])
@pytest.mark.parametrize(
    "reasoning_key",
    [
        "gen_ai.usage.reasoning_tokens",
        "gen_ai.usage.completion_tokens_details.reasoning_tokens",
    ],
)
def test_reasoning_tokens_and_both_system_keys(system_key: str, reasoning_key: str) -> None:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tracer_provider.get_tracer(__name__)
    with tracer.start_as_current_span("chat o4-mini") as span:
        span.set_attributes(
            {
                system_key: "openai",
                "server.address": "api.openai.com",
                "server.port": 443,
                "gen_ai.usage.input_tokens": 1200,
                "gen_ai.usage.output_tokens": 300,
                reasoning_key: 120,
            }
        )
    attributes = dict(cast(Mapping[str, Any], exporter.get_finished_spans()[0].attributes))
    assert (
        attributes.get(SpanAttributes.OPENINFERENCE_SPAN_KIND)
        == OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.get(SpanAttributes.LLM_SYSTEM) == OpenInferenceLLMSystemValues.OPENAI.value
    assert _token_counts(attributes) == {PROMPT: 1200, COMPLETION: 300, REASONING: 120}
