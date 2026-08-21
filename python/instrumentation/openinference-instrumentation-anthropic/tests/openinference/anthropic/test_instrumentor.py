# ruff: noqa: E501
import importlib
import json
import random
import string
from types import ModuleType
from typing import Any, Dict, Optional

import anthropic
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
from anthropic.resources.beta.messages import Messages as BetaMessages
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import (
    ImageBlockParam,
    Message,
    MessageParam,
    RedactedThinkingBlock,
    RedactedThinkingBlockParam,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ThinkingBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
    Usage,
)
from httpx import Response
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._importlib_metadata import entry_points
from pydantic import BaseModel
from respx import MockRouter
from wrapt import BoundFunctionWrapper

from openinference.instrumentation import (
    REDACTED_VALUE,
    OITracer,
    TraceConfig,
    suppress_tracing,
    using_attributes,
)
from openinference.instrumentation.anthropic import AnthropicInstrumentor
from openinference.instrumentation.anthropic._stream import _MessageExtractor
from openinference.instrumentation.anthropic._wrappers import (
    _get_llm_input_messages,
    _get_llm_token_counts,
    _get_output_messages,
)
from openinference.semconv.trace import (
    DocumentAttributes,
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMProviderValues,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)


def _legacy_completions() -> Optional[ModuleType]:
    """
    The legacy Text Completions API and the ``HUMAN_PROMPT``/``AI_PROMPT`` constants
    that went with it were removed in anthropic 1.0.
    """
    try:
        return importlib.import_module("anthropic.resources.completions")
    except ImportError:
        return None


_completions = _legacy_completions()

requires_legacy_completions = pytest.mark.skipif(
    _completions is None,
    reason="anthropic>=1.0 removed the legacy Text Completions API",
)

HUMAN_PROMPT = "\n\nHuman:"
AI_PROMPT = "\n\nAssistant:"


def _get_tool_use_id(message: Message) -> Optional[str]:
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            return block.id
    return None


def assert_json_contains(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        for key, value in expected.items():
            assert key in actual
            assert_json_contains(actual[key], value)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_json_contains(actual_item, expected_item)
        return
    assert actual == expected


def assert_output_value_contains(output_value: str, expected: Any) -> None:
    assert_json_contains(json.loads(output_value), expected)


def _message_json(
    text: str,
    *,
    model: str = "claude-sonnet-4-6",
    message_id: str = "msg_1",
    usage: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A minimal non-streamed assistant message holding a single text block."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage or {"input_tokens": 14, "output_tokens": 11},
    }


FINISH_REASON_USAGE = {"input_tokens": 10, "output_tokens": 5}


def _pop_finish_reason_attributes(
    attributes: Dict[str, Any],
    invocation_parameters: Dict[str, Any],
) -> None:
    """
    Pops everything the finish-reason cases carry apart from LLM_FINISH_REASON, which the
    caller asserts on. Shared by the streamed and non-streamed parametrisations, which
    differ only in their invocation parameters.
    """
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_parameters
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "hello"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
    ) == ("hi")
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == FINISH_REASON_USAGE["input_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == FINISH_REASON_USAGE["output_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == sum(FINISH_REASON_USAGE.values())


def _pop_message_attributes(
    attributes: Dict[str, Any],
    *,
    input_message: str = "hello",
    output_text: str = "hi",
) -> None:
    """
    Pops every attribute a non-streamed message span carries apart from INPUT_VALUE and
    LLM_INVOCATION_PARAMETERS, which the caller asserts on before `assert not attributes`.
    Assumes the response came from `_message_json` with its default usage.
    """
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON) == "end_turn"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == output_text
    )
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 14
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 11
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 25


def _text_message_sse(
    text: str,
    usage: Dict[str, Any],
    *,
    model: str = "claude-sonnet-4-6",
    message_id: str = "msg_1",
    stop_reason: str = "end_turn",
) -> bytes:
    """
    Server-sent events for one streamed assistant message holding a single text block.
    The Messages API and the beta Messages API share this wire format.
    """

    def event(name: str, data: Dict[str, Any]) -> bytes:
        return f"event: {name}\ndata: ".encode() + json.dumps(data).encode() + b"\n\n"

    return b"".join(
        [
            event(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage,
                    },
                },
            ),
            event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
            ),
            event("content_block_stop", {"type": "content_block_stop", "index": 0}),
            event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": usage,
                },
            ),
            event("message_stop", {"type": "message_stop"}),
        ]
    )


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="anthropic"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, AnthropicInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self, setup_anthropic_instrumentation: Any) -> None:
        assert isinstance(AnthropicInstrumentor()._tracer, OITracer)


@requires_legacy_completions
@pytest.mark.vcr
def test_anthropic_instrumentation_completions_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client: Any = Anthropic(api_key="sk-ant-fake")

    prompt = f"{HUMAN_PROMPT} why is the sky blue? respond in five words or less. {AI_PROMPT}"

    stream = client.completions.create(
        model="claude-sonnet-4-6",
        prompt=prompt,
        max_tokens_to_sample=1000,
        stream=True,
    )
    for event in stream:
        print(event.completion)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "completions.create"
    attributes = dict(spans[0].attributes or {})
    print(attributes)

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "completion": " Light scatters blue.",
            "stop": "\n\nHuman:",
            "stop_reason": "stop_sequence",
            "id": "compl_015dfgyiT7JLszAiMbGMtgeG",
            "model": "claude-2.1",
            "type": "completion",
            "log_id": "compl_015dfgyiT7JLszAiMbGMtgeG",
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "stop_sequence"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)

    invocation_params = {"max_tokens_to_sample": 1000, "stream": True}
    assert json.loads(inv_params) == invocation_params
    assert attributes.pop(LLM_OUTPUT_MESSAGES) == " Light scatters blue."
    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_stream_message(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="sk-ant-fake")
    input_message = "What's the capital of France?"
    chat = [{"role": "user", "content": input_message}]
    invocation_params = {"max_tokens": 1024, "stream": True}

    with client.messages.stream(
        max_tokens=1024,
        messages=chat,  # type: ignore
        model="claude-sonnet-4-6",
    ) as stream:
        text = "".join(stream.text_stream)
    assert text == "The capital of France is **Paris**."

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "messages.stream"

    attributes = dict(span.attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    msg_out = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
    assert isinstance(msg_out, str)
    assert "paris" in msg_out.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01GembpbFoc2YxE29Fr2Najf",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "The capital of France is **Paris**.",
                    "type": "text",
                    "parsed_output": None,
                }
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 0,
                },
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 14,
                "output_tokens": 11,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop("llm.token_count.total"), int)

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    raw_inv = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(raw_inv, str)
    assert json.loads(raw_inv) == invocation_params

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_anthropic_instrumentation_async_stream_message(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="sk-ant-fake")
    input_message = "What's the capital of France?"
    chat = [{"role": "user", "content": input_message}]
    invocation_params = {"max_tokens": 1024, "stream": True}

    async with client.messages.stream(
        max_tokens=1024,
        messages=chat,  # type: ignore
        model="claude-sonnet-4-6",
    ) as stream:
        text = "".join([chunk async for chunk in stream.text_stream])
    assert text == "The capital of France is **Paris**."

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "messages.stream"

    attributes = dict(span.attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    msg_out = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
    assert isinstance(msg_out, str)
    assert "paris" in msg_out.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01BUqzFEJ3DSwjUMaBD8QfBm",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "The capital of France is **Paris**.",
                    "type": "text",
                    "parsed_output": None,
                }
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 0,
                },
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 14,
                "output_tokens": 11,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop("llm.token_count.total"), int)

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    raw_inv = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(raw_inv, str)
    assert json.loads(raw_inv) == invocation_params

    assert not attributes


@requires_legacy_completions
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_anthropic_instrumentation_async_completions_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client: Any = AsyncAnthropic(api_key="sk-ant-fake")

    prompt = f"{HUMAN_PROMPT} why is the sky blue? respond in five words or less. {AI_PROMPT}"

    stream = await client.completions.create(
        model="claude-2.1",
        prompt=prompt,
        max_tokens_to_sample=1000,
        stream=True,
    )
    async for event in stream:
        print(event.completion)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "completions.create"
    attributes = dict(spans[0].attributes or {})
    print(attributes)

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "completion": " Light scatters blue.",
            "stop": "\n\nHuman:",
            "stop_reason": "stop_sequence",
            "id": "compl_01Ho8r6LNPQ9EVEAh3vpiUnQ",
            "model": "claude-2.1",
            "type": "completion",
            "log_id": "compl_01Ho8r6LNPQ9EVEAh3vpiUnQ",
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert attributes.pop(LLM_FINISH_REASON, None) == "stop_sequence"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)

    invocation_params = {"max_tokens_to_sample": 1000, "stream": True}
    assert json.loads(inv_params) == invocation_params
    assert attributes.pop(LLM_OUTPUT_MESSAGES) == " Light scatters blue."
    assert not attributes


@requires_legacy_completions
@pytest.mark.vcr
def test_anthropic_instrumentation_completions(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client: Any = Anthropic(api_key="sk-ant-fake")

    invocation_params = {"max_tokens_to_sample": 1000}

    prompt = f"{HUMAN_PROMPT} how does a court case get to the Supreme Court? {AI_PROMPT}"

    client.completions.create(
        model="claude-sonnet-4-6",
        prompt=prompt,
        max_tokens_to_sample=1000,
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "completions.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "compl_01N6jAWfEZtyE338jUQFx9LC",
            "completion": ' A court case can reach the Supreme Court in a few different ways:\n\n1. Appeal from lower courts. Most cases that reach the Supreme Court are appeals from decisions of federal courts of appeals or state supreme courts. Typically there has to be an important constitutional issue or federal law question for the Supreme Court to accept such an appeal.\n\n2. Original jurisdiction cases. The Supreme Court has original jurisdiction over certain types of cases, meaning they can hear the case directly without it coming from a lower court. These include cases between two or more U.S. states or cases involving ambassadors and other diplomats.\n\n3. Certiorari. This is the process by which the Supreme Court selects most of the cases it hears. Parties to a case petition the Court to review the case, and the Court grants "cert" if four of the nine justices agree to hear it. The Court typically grants certiorari in cases that have broad legal impact or important constitutional questions.\n\n4. Certificate from appeals courts. A federal appeals court can also ask the Supreme Court to take a case by granting a certificate of ascertainability. This happens when the appeals court determines there is a critical question of law that requires the Supreme Court\'s review. \n\nSo in most cases, the Supreme Court exercises discretionary review via petitions for certiorari or certificates from lower courts. Its original jurisdiction over certain types of cases also allows some direct access for parties.',
            "model": "claude-2.1",
            "stop_reason": "stop_sequence",
            "type": "completion",
            "stop": "\n\nHuman:",
            "log_id": "compl_01N6jAWfEZtyE338jUQFx9LC",
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_messages(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="sk-ant-fake")
    input_message = "What's the capital of France?"
    system_prompt = "You are a helpful geography assistant."

    invocation_params = {"max_tokens": 1024}

    client.messages.create(
        max_tokens=1024,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-sonnet-4-6",
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "messages.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == system_prompt
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        msg_content := attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
        ),
        str,
    )
    assert "paris" in msg_content.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01BxqRkrCj33q9PDFgWUx6tL",
            "container": None,
            "content": [
                {"citations": None, "text": "The capital of France is **Paris**.", "type": "text"}
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 14,
                "output_tokens": 11,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_messages_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="sk-ant-fake")
    input_message = "Why is the sky blue? Answer in 5 words or less"

    invocation_params = {"max_tokens": 1024, "stream": True}

    stream = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-sonnet-4-6",
        stream=True,
    )

    for event in stream:
        print(event)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "messages.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        msg_content := attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
        ),
        str,
    )
    assert "Sunlight scatters off air molecules." in msg_content
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 21
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 13
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 34

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01VD6x3Z6qzLGuHWS6J7MU86",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "Sunlight scatters off air molecules.",
                    "type": "text",
                    "parsed_output": None,
                }
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 0,
                },
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 21,
                "output_tokens": 13,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_anthropic_instrumentation_async_messages_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="sk-ant-fake")
    input_message = "Why is the sky blue? Answer in 5 words or less"

    invocation_params = {"max_tokens": 1024, "stream": True}

    stream = await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-sonnet-4-6",
        stream=True,
    )

    async for event in stream:
        print(event)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "messages.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        msg_content := attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
        ),
        str,
    )
    assert "Sunlight scatters off air molecules." in msg_content
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 21
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 13
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 34

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01VtWT6cAKHFZxepjCR9Bwk8",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "Sunlight scatters off air molecules.",
                    "type": "text",
                    "parsed_output": None,
                }
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {
                    "ephemeral_1h_input_tokens": 0,
                    "ephemeral_5m_input_tokens": 0,
                },
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 21,
                "output_tokens": 13,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@requires_legacy_completions
@pytest.mark.vcr
async def test_anthropic_instrumentation_async_completions(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client: Any = AsyncAnthropic(api_key="sk-ant-fake")

    invocation_params = {"max_tokens_to_sample": 1000}

    prompt = f"{HUMAN_PROMPT} how does a court case get to the Supreme Court? {AI_PROMPT}"

    await client.completions.create(
        model="claude-sonnet-4-6",
        prompt=prompt,
        max_tokens_to_sample=1000,
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "completions.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "compl_01UXLihn1JiHdBhcGahQv7pe",
            "completion": " A court case can reach the Supreme Court in a few different ways:\n\n1. Appeal from lower courts. Most cases that reach the Supreme Court are appeals from decisions at lower federal courts or state supreme courts. Typically, a party who loses at a lower court level can appeal the decision to the next higher court. After the court of appeals, the next stop is the Supreme Court.\n\n2. Original jurisdiction. The Supreme Court has original jurisdiction over certain types of cases, meaning they can hear the case directly without it coming from a lower court. These mainly include cases between two or more states or certain cases involving ambassadors and public ministers.\n\n3. Writ of certiorari. This a process where a party petitions the Supreme Court to hear an appeal from a lower court. The Supreme Court then has discretion on whether or not it wants to hear the case. Each term, there are thousands of petitions for writ of certiorari, but the court only agrees to hear argument in about 100-150 cases per session. \n\n4. Certificate from lower courts or government. Sometimes a circuit court of appeals can certify a legal issue to the Supreme Court before making a final ruling. Government agencies can also refer cases or issues over which there is some uncertainty or disagreement over the correct legal interpretation.\n\nSo in summary, it's usually an appeals process from lower courts, the court's original jurisdiction, or the Supreme Court agreeing to exercise its discretion in hearing an appeal petition on a disputed legal issue. The type and complexity of cases it hears is very selective.",
            "model": "claude-2.1",
            "stop_reason": "stop_sequence",
            "type": "completion",
            "stop": "\n\nHuman:",
            "log_id": "compl_01UXLihn1JiHdBhcGahQv7pe",
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr
async def test_anthropic_instrumentation_async_messages(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="sk-ant-fake")
    input_message = "What's the capital of France?"

    invocation_params = {"max_tokens": 1024}

    await client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-sonnet-4-6",
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "messages.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        msg_content := attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
        ),
        str,
    )
    assert "paris" in msg_content.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01Hh9cnsgo5riFbYs1zTtC9s",
            "container": None,
            "content": [
                {"citations": None, "text": "The capital of France is **Paris**.", "type": "text"}
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 14,
                "output_tokens": 11,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_multiple_tool_calling(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = anthropic.Anthropic(api_key="sk-ant-fake")

    input_message = (
        "What is the weather like right now in New York?"
        " Also what time is it there? Use necessary tools simultaneously."
    )
    get_weather_tool_schema = ToolParam(
        name="get_weather",
        description="Get the current weather in a given location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                },
            },
            "required": ["location"],
        },
    )
    get_time_tool_schema = ToolParam(
        name="get_time",
        description="Get the current time in a given time zone",
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        },
    )
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[get_weather_tool_schema, get_time_tool_schema],
        messages=[{"role": "user", "content": input_message}],
    )

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "messages.create"
    attributes = dict(spans[0].attributes or {})

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
    assert attributes.pop(LLM_FINISH_REASON, None) == "tool_use"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert isinstance(tool_schema0 := attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}"), str)
    assert json.loads(tool_schema0) == get_weather_tool_schema
    assert isinstance(tool_schema1 := attributes.pop(f"{LLM_TOOLS}.1.{TOOL_JSON_SCHEMA}"), str)
    assert json.loads(tool_schema1) == get_time_tool_schema
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}"), str
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert isinstance(
        attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_ID}"), str
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "get_time"
    )
    assert isinstance(
        attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    # MESSAGE_CONTENTS mirrors tool_use at content position (index 1 = get_weather, 2 = get_time)
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}")
        == "tool_use"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{TOOL_CALL_ID}"), str
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert isinstance(
        attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{MESSAGE_CONTENT_TYPE}")
        == "tool_use"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{TOOL_CALL_ID}"), str
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{TOOL_CALL_FUNCTION_NAME}")
        == "get_time"
    )
    assert isinstance(
        attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_011geMdd2NTwJrvqbfqskQ7r",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "Sure! Let me fetch the current weather and time in New York simultaneously!",
                    "type": "text",
                },
                {
                    "id": "toolu_01VLL6XYAAGrtc7CDpmpKZMB",
                    "caller": {"type": "direct"},
                    "input": {"location": "New York, NY"},
                    "name": "get_weather",
                    "type": "tool_use",
                },
                {
                    "id": "toolu_01FZuC4jLWM67hKreLMKCLRe",
                    "caller": {"type": "direct"},
                    "input": {"timezone": "America/New_York"},
                    "name": "get_time",
                    "type": "tool_use",
                },
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 721,
                "output_tokens": 112,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert isinstance(attributes.pop(OUTPUT_MIME_TYPE), str)
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_multiple_tool_calling_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = anthropic.Anthropic(api_key="sk-ant-fake")

    input_message = (
        "What is the weather like right now in New York?"
        " Also what time is it there? Use necessary tools simultaneously."
    )
    get_weather_tool_schema = ToolParam(
        name="get_weather",
        description="Get the current weather in a given location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                },
            },
            "required": ["location"],
        },
    )
    get_time_tool_schema = ToolParam(
        name="get_time",
        description="Get the current time in a given time zone",
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The IANA time zone name, e.g. America/Los_Angeles",
                }
            },
            "required": ["timezone"],
        },
    )
    stream = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[get_weather_tool_schema, get_time_tool_schema],
        messages=[{"role": "user", "content": input_message}],
        stream=True,
    )
    for event in stream:
        print(event)

    spans = in_memory_span_exporter.get_finished_spans()

    assert spans[0].name == "messages.create"
    attributes = dict(spans[0].attributes or {})

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
    assert attributes.pop(LLM_FINISH_REASON, None) == "tool_use"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert isinstance(tool_schema0 := attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}"), str)
    assert json.loads(tool_schema0) == get_weather_tool_schema
    assert isinstance(tool_schema1 := attributes.pop(f"{LLM_TOOLS}.1.{TOOL_JSON_SCHEMA}"), str)
    assert json.loads(tool_schema1) == get_time_tool_schema
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}"), str
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    get_weather_input_str = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    assert json.loads(get_weather_input_str) == {"location": "New York, NY"}  # type: ignore
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_ID}"), str
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "get_time"
    )
    get_time_input_str = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    json.loads(get_time_input_str) == {"timezone": "America/New_York"}  # type: ignore
    # MESSAGE_CONTENTS mirrors tool_use at content position (index 1 = get_weather, 2 = get_time)
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}")
        == "tool_use"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{TOOL_CALL_ID}"), str
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert isinstance(
        attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{MESSAGE_CONTENT_TYPE}")
        == "tool_use"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{TOOL_CALL_ID}"), str
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{TOOL_CALL_FUNCTION_NAME}")
        == "get_time"
    )
    assert isinstance(
        attributes.pop(
            f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        ),
        str,
    )
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 721
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 113
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 834
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01JqiwuyYfmoZBJx1GLkqxLf",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "I'll check both the current weather and time in New York simultaneously right away!",
                    "type": "text",
                    "parsed_output": None,
                },
                {
                    "id": "toolu_01Mo5Ee5Yb7vrzaxSNS5DVuP",
                    "caller": {"type": "direct"},
                    "input": {"location": "New York, NY"},
                    "name": "get_weather",
                    "type": "tool_use",
                },
                {
                    "id": "toolu_01GDAGw1KUdi1DCPPprMKGHR",
                    "caller": {"type": "direct"},
                    "input": {"timezone": "America/New_York"},
                    "name": "get_time",
                    "type": "tool_use",
                },
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 721,
                "output_tokens": 113,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_image_input_messages_with_stream(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = anthropic.Anthropic(api_key="sk-ant-fake")
    base64_image = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wC="
    image_block = ImageBlockParam(
        type="image",
        source={
            "type": "base64",
            "media_type": "image/png",
            "data": base64_image,
        },
    )
    text_block = TextBlockParam(
        type="text", text="What do you see in this image? Describe it in detail."
    )
    input_messages = [
        MessageParam(
            content=[
                text_block,
                image_block,
            ],
            role="user",
        )
    ]
    system_prompt = [
        TextBlockParam(type="text", text="You are an expert image analyst."),
        TextBlockParam(type="text", text="Always answer concisely."),
    ]
    stream = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        system=system_prompt,
        messages=input_messages,
        stream=True,
    )
    events = [event for event in stream]
    assert len(events) > 0
    spans = in_memory_span_exporter.get_finished_spans()
    assert spans[0].name == "messages.create"
    attributes: Dict[str, Any] = dict(spans[0].attributes or dict())
    assert attributes.pop(LLM_MODEL_NAME) == "claude-3-5-sonnet-20240620"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    # System (list of text blocks) is exposed as a synthetic system message at index 0,
    # with each block indexed under MESSAGE_CONTENTS.
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == "You are an expert image analyst."
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}")
        == "Always answer concisely."
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"),
        str,
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}")
        == "image"
    )
    assert attributes.pop(
        f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
    ).startswith("data:image/png;base64")
    assert isinstance(attributes.pop(f"{LLM_INVOCATION_PARAMETERS}"), str)
    assert attributes.pop(f"{INPUT_MIME_TYPE}") == "application/json"
    assert attributes.pop(f"{OUTPUT_MIME_TYPE}") == "application/json"
    assert isinstance(attributes.pop(f"{INPUT_VALUE}"), str)
    output_value = attributes.pop(f"{OUTPUT_VALUE}")
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_013xrHEn3mecgN2zref6P1is",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "This image shows the iconic Taj Mahal, one of the most famous monuments in the world, located in Agra, India. The majestic white marble mausoleum is perfectly centered in the frame, its distinctive dome and minarets standing out against a clear blue sky.\n\nIn the foreground, there's a long rectangular reflecting pool that leads up to the main building. The water in the pool creates a mirror image of the Taj Mahal, enhancing its beauty and symmetry. On either side of the pool, there are well-manicured green lawns and a row of tall, slender cypress trees, which add to the symmetrical design of the complex.\n\nThe Taj Mahal itself is a stunning example of Mughal architecture. Its central dome is large and bulbous, flanked by four smaller domes. At each corner of the platform on which the mausoleum sits, there are tall, tapering minarets. The entire structure appears to be made of white marble, which gives it a pristine, almost ethereal appearance in the sunlight.\n\nThe scene conveys a sense of serenity, grandeur, and perfect balance. It's a classic view of this UNESCO World Heritage site, capturing the timeless beauty that has made the Taj Mahal one of the most recognizable and admired buildings in the world.",
                    "type": "text",
                    "parsed_output": None,
                }
            ],
            "model": "claude-3-5-sonnet-20240620",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": None,
                "input_tokens": 78,
                "output_tokens": 296,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
    ).startswith("This image shows the iconic Taj Mahal")
    assert attributes.pop(f"{LLM_TOKEN_COUNT_COMPLETION}") == 296
    assert attributes.pop(f"{LLM_TOKEN_COUNT_PROMPT}") == 78
    assert attributes.pop(f"{LLM_TOKEN_COUNT_TOTAL}") == 374
    assert attributes.pop(f"{OPENINFERENCE_SPAN_KIND}") == "LLM"
    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_image_input_messages(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = anthropic.Anthropic(api_key="sk-ant-fake")
    base64_image = "/9j/4AAQSkZJRgABAQAAAQABAAD/2wC="
    image_block = ImageBlockParam(
        type="image",
        source={
            "type": "base64",
            "media_type": "image/png",
            "data": base64_image,
        },
    )
    text_block = TextBlockParam(
        type="text", text="What do you see in this image? Describe it in detail."
    )
    input_messages = [
        MessageParam(
            content=[
                text_block,
                image_block,
            ],
            role="user",
        )
    ]
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620", max_tokens=1024, messages=input_messages
    )
    assert response is not None
    spans = in_memory_span_exporter.get_finished_spans()
    assert spans[0].name == "messages.create"
    attributes: Dict[str, Any] = dict(spans[0].attributes or {})
    assert attributes.pop(LLM_MODEL_NAME) == "claude-3-5-sonnet-20240620"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"),
        str,
    )
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}")
        == "image"
    )
    assert attributes.pop(
        f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
    ).startswith("data:image/png;base64")
    assert isinstance(attributes.pop(f"{LLM_INVOCATION_PARAMETERS}"), str)
    assert attributes.pop(f"{INPUT_MIME_TYPE}") == "application/json"
    assert attributes.pop(f"{OUTPUT_MIME_TYPE}") == "application/json"
    assert isinstance(attributes.pop(f"{INPUT_VALUE}"), str)
    output_value = attributes.pop(f"{OUTPUT_VALUE}")
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01DijAsAzrH5wFcik1mPQjPn",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "This image shows the iconic Taj Mahal, one of the most famous landmarks in the world, located in Agra, India. The majestic white marble mausoleum stands prominently at the end of a long reflecting pool. Its distinctive dome and minarets are perfectly symmetrical and stand out against a clear blue sky.\n\nIn the foreground, we see a long, rectangular water feature that reflects the Taj Mahal, creating a mirror image on its surface. This reflecting pool is lined on both sides by well-manicured green lawns and what appear to be cypress trees, adding to the symmetry and formal garden design.\n\nThe architecture of the Taj Mahal is exquisite, showcasing intricate Islamic design elements. The central dome is large and bulbous, flanked by four smaller domes. At each corner of the main structure stands a tall, slender minaret.\n\nThe entire scene exudes a sense of serenity, grandeur, and timeless beauty. The pristine white of the marble contrasts beautifully with the vibrant green of the gardens and the azure blue of the sky, creating a striking and memorable image that captures the essence of this world-renowned monument.",
                    "type": "text",
                }
            ],
            "model": "claude-3-5-sonnet-20240620",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": None,
                "input_tokens": 78,
                "output_tokens": 263,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"
    ).startswith("This image shows the iconic Taj Mahal")
    assert attributes.pop(f"{LLM_TOKEN_COUNT_COMPLETION}") == 263
    assert attributes.pop(f"{LLM_TOKEN_COUNT_PROMPT}") == 78
    assert attributes.pop(f"{LLM_TOKEN_COUNT_TOTAL}") == 341
    assert attributes.pop(f"{OPENINFERENCE_SPAN_KIND}") == "LLM"
    assert not attributes


@pytest.mark.vcr
@pytest.mark.parametrize(
    "assistant_message",
    (
        pytest.param(
            {
                "content": [
                    TextBlock(
                        text="Certainly! I can help you get the current weather information for"
                        " San Francisco in Fahrenheit. To do this, I'll use the get_weather"
                        " function. Let me fetch that information for you right away.",
                        type="text",
                    ),
                    ToolUseBlock(
                        id="toolu_01KBqpqR73qWGsMaW3vBzEjz",
                        input={"location": "San Francisco, CA", "unit": "fahrenheit"},
                        name="get_weather",
                        type="tool_use",
                    ),
                ],
                "role": "assistant",
            },
            id="with_blocks",
        ),
        pytest.param(
            {
                "content": [
                    TextBlockParam(
                        text="Certainly! I can help you get the current weather information for"
                        " San Francisco in Fahrenheit. To do this, I'll use the get_weather"
                        " function. Let me fetch that information for you right away.",
                        type="text",
                    ),
                    ToolUseBlockParam(
                        id="toolu_01KBqpqR73qWGsMaW3vBzEjz",
                        input={"location": "San Francisco, CA", "unit": "fahrenheit"},
                        name="get_weather",
                        type="tool_use",
                    ),
                ],
                "role": "assistant",
            },
            id="with_block_params",
        ),
    ),
)
def test_anthropic_instrumentation_tool_use_in_input(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
    assistant_message: MessageParam,
) -> None:
    client = anthropic.Anthropic(api_key="sk-ant-fake")
    messages = [
        {"role": "user", "content": "What is the weather like in San Francisco in Fahrenheit?"},
        assistant_message,
        MessageParam(
            content=[
                ToolResultBlockParam(
                    tool_use_id="toolu_01KBqpqR73qWGsMaW3vBzEjz",
                    content='{"weather": "sunny", "temperature": "75"}',
                    type="tool_result",
                    is_error=False,
                )
            ],
            role="user",
        ),
    ]

    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        tools=[
            {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature,"
                            ' either "celsius" or "fahrenheit"',
                        },
                    },
                    "required": ["location"],
                },
            }
        ],
        messages=messages,  # type: ignore
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})

    tool_arguments = '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
    tool_use_id = "toolu_01KBqpqR73qWGsMaW3vBzEjz"

    # The assistant turn carries the tool call both as a message-level tool_calls entry
    # and as a content block.
    assistant = f"{LLM_INPUT_MESSAGES}.1"
    assert attributes.pop(f"{assistant}.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{assistant}.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert (
        attributes.pop(f"{assistant}.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}")
        == tool_arguments
    )
    assert attributes.pop(f"{assistant}.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_ID}") == tool_use_id
    assert attributes.pop(f"{assistant}.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}") == "text"
    assert isinstance(
        attributes.pop(f"{assistant}.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )
    assert attributes.pop(f"{assistant}.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}") == "tool_use"
    assert (
        attributes.pop(f"{assistant}.{MESSAGE_CONTENTS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert (
        attributes.pop(f"{assistant}.{MESSAGE_CONTENTS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}")
        == tool_arguments
    )
    assert attributes.pop(f"{assistant}.{MESSAGE_CONTENTS}.1.{TOOL_CALL_ID}") == tool_use_id

    # The tool result turn keeps the id that ties it back to the call above.
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_CONTENT}")
        == '{"weather": "sunny", "temperature": "75"}'
    )
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "user"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_TOOL_CALL_ID}") == tool_use_id

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
        == "What is the weather like in San Francisco in Fahrenheit?"
    )

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {"max_tokens": 1024}
    assert isinstance(tool_schema := attributes.pop(f"{LLM_TOOLS}.0.{TOOL_JSON_SCHEMA}"), str)
    assert json.loads(tool_schema)["name"] == "get_weather"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert not attributes


@requires_legacy_completions
@pytest.mark.vcr
def test_anthropic_instrumentation_context_attributes_existence(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    session_id = "my-test-session-id"
    user_id = "my-test-user-id"
    metadata = {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }
    tags = ["tag-1", "tag-2"]
    prompt_template = (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )
    prompt_template_version = "v1.0"
    prompt_template_variables = {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }

    client: Any = Anthropic(api_key="sk-ant-fake")

    prompt = f"{HUMAN_PROMPT} how does a court case get to the Supreme Court? {AI_PROMPT}"

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        client.completions.create(
            model="claude-sonnet-4-6",
            prompt=prompt,
            max_tokens_to_sample=1000,
        )

    spans = in_memory_span_exporter.get_finished_spans()

    assert len(spans) == 1
    assert spans[0].name == "completions.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(SESSION_ID) == session_id
    assert attributes.pop(USER_ID) == user_id
    assert isinstance(recorded_metadata := attributes.pop(METADATA), str)
    assert json.loads(recorded_metadata) == metadata
    assert attributes.pop(TAG_TAGS) == tuple(tags)
    assert attributes.pop(LLM_PROMPT_TEMPLATE) == prompt_template
    assert attributes.pop(LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
    assert isinstance(recorded_variables := attributes.pop(LLM_PROMPT_TEMPLATE_VARIABLES), str)
    assert json.loads(recorded_variables) == prompt_template_variables

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {"max_tokens_to_sample": 1000}
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert not attributes


def test_anthropic_instrumentation_messages_context_attributes_existence(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """Context attributes propagate onto messages spans on every supported SDK version."""
    session_id = "my-test-session-id"
    user_id = "my-test-user-id"
    metadata = {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }
    tags = ["tag-1", "tag-2"]
    prompt_template = (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )
    prompt_template_version = "v1.0"
    prompt_template_variables = {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }

    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(status_code=200, json=_message_json("hi"))
    )
    client = Anthropic(api_key="sk-ant-fake")

    with using_attributes(
        session_id=session_id,
        user_id=user_id,
        metadata=metadata,
        tags=tags,
        prompt_template=prompt_template,
        prompt_template_version=prompt_template_version,
        prompt_template_variables=prompt_template_variables,
    ):
        client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": "hello"}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "messages.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(SESSION_ID) == session_id
    assert attributes.pop(USER_ID) == user_id
    assert isinstance(recorded_metadata := attributes.pop(METADATA), str)
    assert json.loads(recorded_metadata) == metadata
    assert attributes.pop(TAG_TAGS) == tuple(tags)
    assert attributes.pop(LLM_PROMPT_TEMPLATE) == prompt_template
    assert attributes.pop(LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
    assert isinstance(recorded_variables := attributes.pop(LLM_PROMPT_TEMPLATE_VARIABLES), str)
    assert json.loads(recorded_variables) == prompt_template_variables

    _pop_message_attributes(attributes)
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {"max_tokens": 1000}
    assert isinstance(attributes.pop(INPUT_VALUE), str)

    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_messages_token_counts(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="sk-ant-fake")
    random_1024_token = "".join(random.choices(string.ascii_letters + string.digits, k=2000))
    novel_text = """Full Text of Novel <Pride and Prejudice>""" + random_1024_token
    client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.\n",
            },
            {
                "type": "text",
                "text": novel_text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {"role": "user", "content": "Analyze the major themes in 'Pride and Prejudice'."}
        ],
    )
    client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": "You are an AI assistant tasked with analyzing literary works.\n",
            },
            {
                "type": "text",
                "text": novel_text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {"role": "user", "content": "Analyze the major themes in 'Pride and Prejudice'."}
        ],
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 2
    s1, s2 = spans
    att1 = dict(s1.attributes or {})
    att2 = dict(s2.attributes or {})
    # Two requests have identical requests/prompts
    assert att1.pop(LLM_TOKEN_COUNT_PROMPT) == att2.pop(LLM_TOKEN_COUNT_PROMPT)
    # first request's cache write is 2nd request's cache read
    assert (
        att1.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE)
        == att2.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ)
        == 1733
    )
    # first request doesn't hit cache
    assert LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ not in att1
    # second request doesn't write cache
    assert LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE not in att2

    # Both spans otherwise carry the same shape: a cached system prompt split into two
    # content blocks, then the user turn.
    for attributes in (att1, att2):
        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
        assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
        assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
        assert attributes.pop(LLM_MODEL_NAME) == "claude-3-7-sonnet-20250219"
        assert attributes.pop(LLM_FINISH_REASON) == "end_turn"
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params) == {"max_tokens": 2048}

        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "system"
        for index in (0, 1):
            prefix = f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.{index}"
            assert attributes.pop(f"{prefix}.{MESSAGE_CONTENT_TYPE}") == "text"
            assert isinstance(attributes.pop(f"{prefix}.{MESSAGE_CONTENT_TEXT}"), str)
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "user"
        assert (
            attributes.pop(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENT}")
            == "Analyze the major themes in 'Pride and Prejudice'."
        )

        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
            == "text"
        )
        assert isinstance(
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"),
            str,
        )

        assert isinstance(attributes.pop(INPUT_VALUE), str)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(attributes.pop(OUTPUT_VALUE), str)
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
        assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

        assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_messages_parse(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    class Capital(BaseModel):
        city: str
        country: str

    client = Anthropic(api_key="sk-ant-fake")
    input_message = "What is the capital of France? Respond with the city and country."

    result = client.messages.parse(
        max_tokens=256,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
        output_format=Capital,
    )
    parsed = result.content[0].parsed_output  # type: ignore[union-attr]
    assert parsed is not None
    assert parsed.city.lower() == "paris"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "messages.parse"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_017pC17fmFPUhGb5UENdPKqG",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": '{"city":"Paris","country":"France"}',
                    "type": "text",
                    "parsed_output": {"city": "Paris", "country": "France"},
                }
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 210,
                "output_tokens": 12,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    raw_inv_params = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(raw_inv_params, str)
    inv_params = json.loads(raw_inv_params)
    assert inv_params == {
        "max_tokens": 256,
        "output_config": {
            "format": {
                "schema": {
                    "additionalProperties": False,
                    "properties": {
                        "city": {"title": "City", "type": "string"},
                        "country": {"title": "Country", "type": "string"},
                    },
                    "required": ["city", "country"],
                    "title": "Capital",
                    "type": "object",
                },
                "type": "json_schema",
            }
        },
    }

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_anthropic_instrumentation_async_messages_parse(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    class Capital(BaseModel):
        city: str
        country: str

    client = AsyncAnthropic(api_key="sk-ant-fake")
    input_message = "What is the capital of France? Respond with the city and country."

    result = await client.messages.parse(
        max_tokens=256,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
        output_format=Capital,
    )
    parsed = result.content[0].parsed_output  # type: ignore[union-attr]
    assert parsed is not None
    assert parsed.city.lower() == "paris"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "messages.parse"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01UxkoYKRxHPYTUYkGic5teK",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": '{"city":"Paris","country":"France"}',
                    "type": "text",
                    "parsed_output": {"city": "Paris", "country": "France"},
                }
            ],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 210,
                "output_tokens": 12,
                "server_tool_use": None,
                "service_tier": "standard",
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    raw_inv_params = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(raw_inv_params, str)
    inv_params = json.loads(raw_inv_params)
    assert inv_params == {
        "max_tokens": 256,
        "output_config": {
            "format": {
                "schema": {
                    "additionalProperties": False,
                    "properties": {
                        "city": {"title": "City", "type": "string"},
                        "country": {"title": "Country", "type": "string"},
                    },
                    "required": ["city", "country"],
                    "title": "Capital",
                    "type": "object",
                },
                "type": "json_schema",
            }
        },
    }

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_beta_messages_parse(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    class Capital(BaseModel):
        city: str
        country: str

    client = Anthropic(api_key="sk-ant-fake")
    input_message = "What is the capital of France? Respond with the city and country."

    result = client.beta.messages.parse(
        max_tokens=256,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
        output_format=Capital,
    )
    parsed = result.content[0].parsed_output  # type: ignore[union-attr]
    assert parsed is not None
    assert parsed.city.lower() == "paris"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "beta.messages.parse"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01CiA3YpvhgJbxvaoofq8Pri",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": '{"city":"Paris","country":"France"}',
                    "type": "text",
                    "parsed_output": {"city": "Paris", "country": "France"},
                }
            ],
            "context_management": None,
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 210,
                "iterations": None,
                "output_tokens": 12,
                "server_tool_use": None,
                "service_tier": "standard",
                "speed": None,
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    raw_inv_params = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(raw_inv_params, str)
    inv_params = json.loads(raw_inv_params)
    assert inv_params == {
        "max_tokens": 256,
        "output_config": {
            "format": {
                "schema": {
                    "additionalProperties": False,
                    "properties": {
                        "city": {"title": "City", "type": "string"},
                        "country": {"title": "Country", "type": "string"},
                    },
                    "required": ["city", "country"],
                    "title": "Capital",
                    "type": "object",
                },
                "type": "json_schema",
            }
        },
    }

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_anthropic_instrumentation_async_beta_messages_parse(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    class Capital(BaseModel):
        city: str
        country: str

    client = AsyncAnthropic(api_key="sk-ant-fake")
    input_message = "What is the capital of France? Respond with the city and country."

    result = await client.beta.messages.parse(
        max_tokens=256,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
        output_format=Capital,
    )
    parsed = result.content[0].parsed_output  # type: ignore[union-attr]
    assert parsed is not None
    assert parsed.city.lower() == "paris"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "beta.messages.parse"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01YLp4hqTXinnBRQ6MMipuy9",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": '{"city":"Paris","country":"France"}',
                    "type": "text",
                    "parsed_output": {"city": "Paris", "country": "France"},
                }
            ],
            "context_management": None,
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 210,
                "iterations": None,
                "output_tokens": 12,
                "server_tool_use": None,
                "service_tier": "standard",
                "speed": None,
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    raw_inv_params = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(raw_inv_params, str)
    inv_params = json.loads(raw_inv_params)
    assert inv_params == {
        "max_tokens": 256,
        "output_config": {
            "format": {
                "schema": {
                    "additionalProperties": False,
                    "properties": {
                        "city": {"title": "City", "type": "string"},
                        "country": {"title": "Country", "type": "string"},
                    },
                    "required": ["city", "country"],
                    "title": "Capital",
                    "type": "object",
                },
                "type": "json_schema",
            }
        },
    }

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert not attributes


def test_anthropic_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

    if _completions is not None:
        assert isinstance(_completions.Completions.create, BoundFunctionWrapper)
        assert isinstance(_completions.AsyncCompletions.create, BoundFunctionWrapper)

    assert isinstance(Messages.create, BoundFunctionWrapper)
    assert isinstance(AsyncMessages.create, BoundFunctionWrapper)
    assert isinstance(Messages.stream, BoundFunctionWrapper)
    assert isinstance(AsyncMessages.stream, BoundFunctionWrapper)
    assert isinstance(Messages.parse, BoundFunctionWrapper)
    assert isinstance(AsyncMessages.parse, BoundFunctionWrapper)

    assert isinstance(BetaMessages.create, BoundFunctionWrapper)
    assert isinstance(AsyncBetaMessages.create, BoundFunctionWrapper)
    assert isinstance(BetaMessages.stream, BoundFunctionWrapper)
    assert isinstance(AsyncBetaMessages.stream, BoundFunctionWrapper)
    assert isinstance(BetaMessages.parse, BoundFunctionWrapper)
    assert isinstance(AsyncBetaMessages.parse, BoundFunctionWrapper)

    AnthropicInstrumentor().uninstrument()

    if _completions is not None:
        assert not isinstance(_completions.Completions.create, BoundFunctionWrapper)
        assert not isinstance(_completions.AsyncCompletions.create, BoundFunctionWrapper)

    assert not isinstance(Messages.create, BoundFunctionWrapper)
    assert not isinstance(AsyncMessages.create, BoundFunctionWrapper)
    assert not isinstance(Messages.stream, BoundFunctionWrapper)
    assert not isinstance(AsyncMessages.stream, BoundFunctionWrapper)
    assert not isinstance(Messages.parse, BoundFunctionWrapper)
    assert not isinstance(AsyncMessages.parse, BoundFunctionWrapper)

    assert not isinstance(BetaMessages.create, BoundFunctionWrapper)
    assert not isinstance(AsyncBetaMessages.create, BoundFunctionWrapper)
    assert not isinstance(BetaMessages.stream, BoundFunctionWrapper)
    assert not isinstance(AsyncBetaMessages.stream, BoundFunctionWrapper)
    assert not isinstance(BetaMessages.parse, BoundFunctionWrapper)
    assert not isinstance(AsyncBetaMessages.parse, BoundFunctionWrapper)


# Ensure we're using the common OITracer from common openinference-instrumentation pkg
def test_oitracer(
    setup_anthropic_instrumentation: Any,
) -> None:
    assert isinstance(AnthropicInstrumentor()._tracer, OITracer)


@pytest.mark.vcr
def test_anthropic_instrumentation_beta_messages_create(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """Test instrumentation for beta.messages.create() method."""
    client = Anthropic(api_key="sk-ant-fake")
    input_message = (
        "Extract the key information from: The meeting is scheduled for March 15th at 2 PM."
    )
    invocation_params = {"max_tokens": 1024}

    client.beta.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "beta.messages.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_01CTGDX2snWfHvBwB14u8Y8P",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "Here is the key information extracted:\n\n- **Event:** Meeting\n- **Date:** March 15th\n- **Time:** 2:00 PM",
                    "type": "text",
                }
            ],
            "context_management": None,
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 28,
                "iterations": None,
                "output_tokens": 36,
                "server_tool_use": None,
                "service_tier": "standard",
                "speed": None,
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_anthropic_instrumentation_async_beta_messages_create(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """Test instrumentation for async beta.messages.create() method."""
    client = AsyncAnthropic(api_key="sk-ant-fake")
    input_message = (
        "Extract the key information from: The meeting is scheduled for March 15th at 2 PM."
    )
    invocation_params = {"max_tokens": 1024}

    await client.beta.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "beta.messages.create"
    attributes = dict(spans[0].attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_015FWiTX6PfnwN4UdLKKAEar",
            "container": None,
            "content": [
                {
                    "citations": None,
                    "text": "Here is the key information extracted:\n\n- **Event:** Meeting\n- **Date:** March 15th\n- **Time:** 2:00 PM",
                    "type": "text",
                }
            ],
            "context_management": None,
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "type": "message",
            "usage": {
                "cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "inference_geo": "global",
                "input_tokens": 28,
                "iterations": None,
                "output_tokens": 36,
                "server_tool_use": None,
                "service_tier": "standard",
                "speed": None,
            },
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert isinstance(
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"), str
    )

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_TOTAL), int)

    assert not attributes


BETA_STREAM_USAGE = {"input_tokens": 14, "output_tokens": 11}
BETA_STREAM_TEXT = "The capital of France is Paris."


def _assert_beta_stream_attributes(
    span: Any,
    span_name: str,
    input_message: str,
    invocation_params: Dict[str, Any],
) -> None:
    """Every beta streaming path must record the same span shape."""
    assert span.name == span_name
    attributes = dict(span.attributes or {})

    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC

    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == input_message
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == BETA_STREAM_TEXT
    )

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(
        output_value,
        {
            "id": "msg_1",
            "content": [{"text": BETA_STREAM_TEXT, "type": "text"}],
            "model": "claude-sonnet-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
        },
    )
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == BETA_STREAM_USAGE["input_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == BETA_STREAM_USAGE["output_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == sum(BETA_STREAM_USAGE.values())

    assert not attributes


def test_anthropic_instrumentation_beta_messages_stream(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """beta.messages.stream() records one span when the stream is exhausted."""
    input_message = "What's the capital of France?"
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            status_code=200,
            content=_text_message_sse(BETA_STREAM_TEXT, BETA_STREAM_USAGE),
        )
    )
    client = Anthropic(api_key="sk-ant-fake")

    with client.beta.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
    ) as stream:
        assert "".join(stream.text_stream) == BETA_STREAM_TEXT

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_beta_stream_attributes(
        spans[0],
        "beta.messages.stream",
        input_message,
        {"max_tokens": 1024, "stream": True},
    )


async def test_anthropic_instrumentation_async_beta_messages_stream(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """The async beta.messages.stream() records the same span as the sync one."""
    input_message = "What's the capital of France?"
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            status_code=200,
            content=_text_message_sse(BETA_STREAM_TEXT, BETA_STREAM_USAGE),
        )
    )
    client = AsyncAnthropic(api_key="sk-ant-fake")

    text = ""
    async with client.beta.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
    ) as stream:
        async for chunk in stream.text_stream:
            text += chunk
    assert text == BETA_STREAM_TEXT

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_beta_stream_attributes(
        spans[0],
        "beta.messages.stream",
        input_message,
        {"max_tokens": 1024, "stream": True},
    )


def test_anthropic_instrumentation_beta_messages_create_streaming(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """beta.messages.create(stream=True) ends its span when iteration completes."""
    input_message = "What's the capital of France?"
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            status_code=200,
            content=_text_message_sse(BETA_STREAM_TEXT, BETA_STREAM_USAGE),
        )
    )
    client = Anthropic(api_key="sk-ant-fake")

    stream = client.beta.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
        stream=True,
    )
    # The span must stay open until the caller drains the stream.
    assert not in_memory_span_exporter.get_finished_spans()
    for _ in stream:
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_beta_stream_attributes(
        spans[0],
        "beta.messages.create",
        input_message,
        {"max_tokens": 1024, "stream": True},
    )


async def test_anthropic_instrumentation_async_beta_messages_create_streaming(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """The async beta.messages.create(stream=True) records the same span as the sync one."""
    input_message = "What's the capital of France?"
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            status_code=200,
            content=_text_message_sse(BETA_STREAM_TEXT, BETA_STREAM_USAGE),
        )
    )
    client = AsyncAnthropic(api_key="sk-ant-fake")

    stream = await client.beta.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": input_message}],
        model="claude-sonnet-4-6",
        stream=True,
    )
    assert not in_memory_span_exporter.get_finished_spans()
    async for _ in stream:
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    _assert_beta_stream_attributes(
        spans[0],
        "beta.messages.create",
        input_message,
        {"max_tokens": 1024, "stream": True},
    )


def test_not_given_parameters_are_omitted_from_attributes(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    A caller may pass the SDK's sentinels to leave parameters unset. Those must not
    reach a span: anthropic>=1.0 renders `Omit` as a heap address, which would make
    input.value differ between otherwise identical runs.
    """
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(status_code=200, json=_message_json("hi"))
    )
    # Typed as Any because the two SDK versions disagree on which sentinel each
    # parameter accepts: anthropic<1.0 types them as NotGiven, anthropic>=1.0 as Omit.
    # Both must be filtered at runtime, so the test passes one of each.
    client: Any = Anthropic(api_key="sk-ant-fake")

    client.beta.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": "hello"}],
        system=anthropic.Omit(),
        stop_sequences=anthropic.NotGiven(),
        # Nested one level down, which the top-level filter alone would miss.
        metadata={"user_id": anthropic.Omit()},
    )

    attributes = dict(in_memory_span_exporter.get_finished_spans()[0].attributes or {})
    rendered = str(attributes)
    _pop_message_attributes(attributes)

    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {"max_tokens": 64, "metadata": {}}
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert json.loads(input_value) == {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {},
    }

    assert not attributes

    for sentinel in ("NotGiven", "NOT_GIVEN", "Omit", "not_given"):
        assert sentinel not in rendered


def test_request_options_are_omitted_from_attributes(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    Per-request transport options are not model parameters and must not be recorded.
    extra_headers is the SDK's documented way to override auth for one call, so
    recording it would put a credential in input.value.
    """
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(status_code=200, json=_message_json("hi"))
    )
    client = Anthropic(api_key="sk-ant-fake")

    client.beta.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": "hello"}],
        extra_headers={"x-api-key": "sk-ant-super-secret"},
        extra_query={"trace": "no"},
        extra_body={"internal": "no"},
        timeout=30.0,
    )

    attributes = dict(in_memory_span_exporter.get_finished_spans()[0].attributes or {})
    rendered = str(attributes)
    _pop_message_attributes(attributes)

    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {"max_tokens": 64}
    assert isinstance(input_value := attributes.pop(INPUT_VALUE), str)
    assert set(json.loads(input_value)) == {"model", "max_tokens", "messages"}

    assert not attributes

    assert "sk-ant-super-secret" not in rendered


def test_suppress_tracing_emits_no_spans(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """suppress_tracing must stop both the plain and the streaming beta paths."""
    route = respx_mock.post("https://api.anthropic.com/v1/messages")
    client = Anthropic(api_key="sk-ant-fake")
    kwargs: Dict[str, Any] = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }

    with suppress_tracing():
        route.mock(return_value=Response(status_code=200, json=_message_json("hi")))
        client.beta.messages.create(**kwargs)

        route.mock(
            return_value=Response(
                status_code=200, content=_text_message_sse("hi", BETA_STREAM_USAGE)
            )
        )
        with client.beta.messages.stream(**kwargs) as stream:
            "".join(stream.text_stream)

    assert in_memory_span_exporter.get_finished_spans() == ()

    # Tracing resumes once the suppression scope exits.
    route.mock(return_value=Response(status_code=200, json=_message_json("hi")))
    client.beta.messages.create(**kwargs)
    assert len(in_memory_span_exporter.get_finished_spans()) == 1


def test_trace_config_hides_inputs(
    respx_mock: MockRouter,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """hide_inputs must redact input.value and drop the input messages."""
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(status_code=200, json=_message_json("hi"))
    )
    AnthropicInstrumentor().instrument(
        tracer_provider=tracer_provider, config=TraceConfig(hide_inputs=True)
    )
    try:
        Anthropic(api_key="sk-ant-fake").beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=64,
            messages=[{"role": "user", "content": "hello"}],
        )
    finally:
        AnthropicInstrumentor().uninstrument()

    attributes = dict(in_memory_span_exporter.get_finished_spans()[0].attributes or {})

    # The input is redacted, and its mime type and messages are dropped entirely.
    assert attributes.pop(INPUT_VALUE) == REDACTED_VALUE
    assert INPUT_MIME_TYPE not in attributes
    assert not [key for key in attributes if key.startswith(LLM_INPUT_MESSAGES)]

    # Everything else, including the output, is untouched.
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {"max_tokens": 64}
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
        == "text"
    )
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == "hi"
    )
    assert isinstance(attributes.pop(OUTPUT_VALUE), str)
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 14
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 11
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 25

    assert not attributes


def test_trace_config_hides_outputs_on_beta_stream(
    respx_mock: MockRouter,
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    """hide_outputs must reach the streaming path, which finishes its span later."""
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(status_code=200, content=_text_message_sse("hi", BETA_STREAM_USAGE))
    )
    AnthropicInstrumentor().instrument(
        tracer_provider=tracer_provider, config=TraceConfig(hide_outputs=True)
    )
    try:
        client = Anthropic(api_key="sk-ant-fake")
        with client.beta.messages.stream(
            model="claude-sonnet-4-6",
            max_tokens=64,
            messages=[{"role": "user", "content": "hello"}],
        ) as stream:
            "".join(stream.text_stream)
    finally:
        AnthropicInstrumentor().uninstrument()

    attributes = dict(in_memory_span_exporter.get_finished_spans()[0].attributes or {})

    # The output is redacted, and its mime type and messages are dropped entirely.
    assert attributes.pop(OUTPUT_VALUE) == REDACTED_VALUE
    assert OUTPUT_MIME_TYPE not in attributes
    assert not [key for key in attributes if key.startswith(LLM_OUTPUT_MESSAGES)]

    # Everything else, including the input, is untouched.
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_FINISH_REASON) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {"max_tokens": 64, "stream": True}
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "hello"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == BETA_STREAM_USAGE["input_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == BETA_STREAM_USAGE["output_tokens"]
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == sum(BETA_STREAM_USAGE.values())

    assert not attributes


def test_api_error_ends_span_with_error_status(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """A failed request must still close its span, marked as an error."""
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(status_code=500, json={"type": "error", "error": {"message": "boom"}})
    )
    client = Anthropic(api_key="sk-ant-fake", max_retries=0)

    with pytest.raises(anthropic.APIStatusError):
        client.beta.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=64,
            messages=[{"role": "user", "content": "hello"}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "beta.messages.create"
    assert span.status.status_code is StatusCode.ERROR
    assert span.events, "the exception must be recorded on the span"

    # Inputs are recorded even though the call failed. Nothing about the response is.
    attributes = dict(span.attributes or {})
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "hello"
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == {"max_tokens": 64}

    assert not attributes


def test_get_output_messages_with_thinking_block() -> None:
    message = Message(
        id="msg_thinking",
        content=[
            ThinkingBlock(
                type="thinking",
                thinking="Let me work through this. The capital of France is Paris.",
                signature="EuYBCkQYAiJA...",
            ),
            TextBlock(type="text", text="Paris."),
        ],
        model="claude-opus-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    attributes = dict(_get_output_messages(message))

    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}"] == "assistant"
    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert (
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"]
        == "Let me work through this. The capital of France is Paris."
    )
    assert (
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_SIGNATURE}"]
        == "EuYBCkQYAiJA..."
    )
    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}"] == (
        "text"
    )
    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}"] == (
        "Paris."
    )

    # message_content.id must never be emitted for thinking/redacted_thinking blocks
    assert not any(key.endswith("message_content.id") for key in attributes)


def test_get_output_messages_with_redacted_thinking_block() -> None:
    message = Message(
        id="msg_redacted_thinking",
        content=[
            RedactedThinkingBlock(
                type="redacted_thinking",
                data="EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2...",
            ),
            TextBlock(type="text", text="Paris."),
        ],
        model="claude-opus-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    attributes = dict(_get_output_messages(message))

    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert (
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_DATA}"]
        == "EmwKAhgBEgy3va3pzix/LafPsn4aDFIT2..."
    )
    assert f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}" not in attributes
    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}"] == (
        "text"
    )

    # message_content.id must never be emitted for thinking/redacted_thinking blocks
    assert not any(key.endswith("message_content.id") for key in attributes)


def test_message_extractor_with_thinking_and_redacted_thinking_blocks() -> None:
    """Streaming responses must capture reasoning fields post-accumulation."""
    snapshot = Message(
        id="msg_stream_thinking",
        content=[
            ThinkingBlock(
                type="thinking",
                thinking="Reasoning about the capital of France...",
                signature="streamed-signature",
            ),
            RedactedThinkingBlock(
                type="redacted_thinking",
                data="streamed-redacted-data",
            ),
            TextBlock(type="text", text="Paris."),
        ],
        model="claude-opus-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=10, output_tokens=20),
    )

    attributes = dict(_MessageExtractor(snapshot).get_attributes())

    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert (
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"]
        == "Reasoning about the capital of France..."
    )
    assert (
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_SIGNATURE}"]
        == "streamed-signature"
    )

    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert (
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_DATA}"]
        == "streamed-redacted-data"
    )
    assert f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}" not in attributes

    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{MESSAGE_CONTENT_TYPE}"] == (
        "text"
    )
    assert (
        attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.2.{MESSAGE_CONTENT_TEXT}"]
        == "Paris."
    )


@pytest.mark.parametrize(
    "cache_read,cache_write",
    [(512, 1733), (512, 0), (0, 1733), (0, 0)],
)
def test_message_extractor_records_cache_token_details(
    cache_read: int,
    cache_write: int,
) -> None:
    """Streaming must break cache tokens out, not only fold them into the prompt total."""
    snapshot = Message(
        id="msg_stream_cache",
        content=[TextBlock(type="text", text="Paris.")],
        model="claude-opus-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(
            input_tokens=10,
            output_tokens=20,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_write,
        ),
    )

    attributes = dict(_MessageExtractor(snapshot).get_attributes())

    # The prompt total counts fresh, read and written tokens, as on the non-streaming path.
    assert attributes[LLM_TOKEN_COUNT_PROMPT] == 10 + cache_read + cache_write
    # A zero count is omitted rather than emitted as 0, matching _get_llm_token_counts.
    assert attributes.get(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == (cache_read or None)
    assert attributes.get(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) == (cache_write or None)


@pytest.mark.parametrize(
    "cache_read,cache_write",
    [(0, 0), (512, 1733)],
)
def test_token_count_total_matches_across_paths(
    cache_read: int,
    cache_write: int,
) -> None:
    """`total` must not depend on whether the caller asked for streaming (#3490).

    Anthropic's Usage carries no total field, so both paths derive it. Comparing the two
    attribute producers directly keeps them from drifting apart again.
    """
    usage = Usage(
        input_tokens=10,
        output_tokens=20,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )
    snapshot = Message(
        id="msg_total",
        content=[TextBlock(type="text", text="Paris.")],
        model="claude-opus-4-6",
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=usage,
    )

    non_streaming = dict(_get_llm_token_counts(usage))
    streaming = dict(_MessageExtractor(snapshot).get_attributes())

    expected_total = 10 + cache_read + cache_write + 20
    assert non_streaming[LLM_TOKEN_COUNT_TOTAL] == expected_total
    assert streaming[LLM_TOKEN_COUNT_TOTAL] == expected_total
    # total is the sum of the two counts it summarizes, on both paths.
    assert expected_total == (
        non_streaming[LLM_TOKEN_COUNT_PROMPT] + non_streaming[LLM_TOKEN_COUNT_COMPLETION]
    )


def test_token_count_total_omitted_when_all_counts_are_zero() -> None:
    """A zero total is skipped rather than emitted as 0, like the other counts."""
    usage = Usage(input_tokens=0, output_tokens=0)
    assert LLM_TOKEN_COUNT_TOTAL not in dict(_get_llm_token_counts(usage))


def test_cache_token_details_match_between_streaming_and_non_streaming(
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """The same usage served two ways must produce the same token attributes."""
    usage = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation_input_tokens": 1733,
        "cache_read_input_tokens": 512,
    }
    sse = _text_message_sse("hi", usage)
    route = respx_mock.post("https://api.anthropic.com/v1/messages")
    client = Anthropic(api_key="sk-ant-fake")
    kwargs: Dict[str, Any] = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "hello"}],
    }

    route.mock(
        return_value=Response(
            status_code=200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": usage,
            },
        )
    )
    client.messages.create(**kwargs)

    route.mock(return_value=Response(status_code=200, content=sse))
    for _ in client.messages.create(stream=True, **kwargs):
        pass

    route.mock(return_value=Response(status_code=200, content=sse))
    with client.messages.stream(**kwargs) as stream:
        for _ in stream:
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert [span.name for span in spans] == [
        "messages.create",
        "messages.create",
        "messages.stream",
    ]

    for span in spans:
        attributes = dict(span.attributes or {})

        # The point of the test: identical usage must yield identical token attributes
        # whichever way it was served.
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 2255
        assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 5
        assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 2260
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) == 512
        assert attributes.pop(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) == 1733

        assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
        assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
        assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
        assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
        assert attributes.pop(LLM_FINISH_REASON) == "end_turn"
        assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
        assert json.loads(inv_params).get("max_tokens") == 1000
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}") == "hello"
        assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
        assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}")
            == "text"
        )
        assert (
            attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
            == "hi"
        )
        assert isinstance(attributes.pop(INPUT_VALUE), str)
        assert attributes.pop(INPUT_MIME_TYPE) == JSON
        assert isinstance(attributes.pop(OUTPUT_VALUE), str)
        assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

        # Nothing left over, so message_content.id can never have been emitted for a
        # thinking or redacted_thinking block.
        assert not attributes


@pytest.mark.parametrize(
    "thinking_block, redacted_thinking_block",
    (
        pytest.param(
            ThinkingBlockParam(
                type="thinking",
                thinking="Reasoning about the request...",
                signature="input-signature",
            ),
            RedactedThinkingBlockParam(
                type="redacted_thinking",
                data="input-redacted-data",
            ),
            id="block_params",
        ),
        pytest.param(
            {
                "type": "thinking",
                "thinking": "Reasoning about the request...",
                "signature": "input-signature",
            },
            {
                "type": "redacted_thinking",
                "data": "input-redacted-data",
            },
            id="dicts",
        ),
    ),
)
def test_get_llm_input_messages_with_thinking_blocks(
    thinking_block: Any,
    redacted_thinking_block: Any,
) -> None:
    """Reasoning blocks round-tripped back as assistant input must surface in
    llm.input_messages, preserving block order."""
    messages: list[MessageParam] = [
        {"role": "user", "content": "What is the capital of France?"},
        {
            "role": "assistant",
            "content": [
                thinking_block,
                redacted_thinking_block,
                TextBlockParam(type="text", text="Paris."),
            ],
        },
    ]

    attributes = dict(_get_llm_input_messages(messages))

    assert attributes[f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}"] == "assistant"
    assert attributes[f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert (
        attributes[f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}"]
        == "Reasoning about the request..."
    )
    assert (
        attributes[f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_SIGNATURE}"]
        == "input-signature"
    )

    assert attributes[f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TYPE}"] == (
        "reasoning"
    )
    assert (
        attributes[f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_DATA}"]
        == "input-redacted-data"
    )
    assert f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.1.{MESSAGE_CONTENT_TEXT}" not in attributes

    assert attributes[f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.2.{MESSAGE_CONTENT_TYPE}"] == (
        "text"
    )
    assert (
        attributes[f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_CONTENTS}.2.{MESSAGE_CONTENT_TEXT}"]
        == "Paris."
    )

    # message_content.id must never be emitted for thinking/redacted_thinking blocks
    assert not any(key.endswith("message_content.id") for key in attributes)


@pytest.mark.parametrize(
    "stop_reason",
    ["end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"],
)
def test_finish_reason_values_messages_create(
    stop_reason: str,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    message = _message_json("hi", message_id="msg_test123", usage=FINISH_REASON_USAGE)
    message["stop_reason"] = stop_reason
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(status_code=200, json=message)
    )
    client = Anthropic(api_key="sk-ant-fake")
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": "hello"}],
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes.pop(LLM_FINISH_REASON) == stop_reason
    _pop_finish_reason_attributes(attributes, {"max_tokens": 1000})
    assert not attributes


@pytest.mark.parametrize(
    "stop_reason",
    ["end_turn", "max_tokens", "tool_use"],
)
def test_finish_reason_values_messages_create_streaming(
    stop_reason: str,
    respx_mock: MockRouter,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=Response(
            status_code=200,
            content=_text_message_sse("hi", FINISH_REASON_USAGE, stop_reason=stop_reason),
        )
    )
    client = Anthropic(api_key="sk-ant-fake")
    stream = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    for _ in stream:
        pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes.pop(LLM_FINISH_REASON) == stop_reason
    _pop_finish_reason_attributes(attributes, {"max_tokens": 1000, "stream": True})
    assert not attributes


CHAIN = OpenInferenceSpanKindValues.CHAIN
LLM = OpenInferenceSpanKindValues.LLM
RETRIEVER = OpenInferenceSpanKindValues.RETRIEVER

JSON = OpenInferenceMimeTypeValues.JSON.value
TEXT = OpenInferenceMimeTypeValues.TEXT.value

DOCUMENT_CONTENT = DocumentAttributes.DOCUMENT_CONTENT
DOCUMENT_ID = DocumentAttributes.DOCUMENT_ID
DOCUMENT_METADATA = DocumentAttributes.DOCUMENT_METADATA
EMBEDDING_EMBEDDINGS = SpanAttributes.EMBEDDING_EMBEDDINGS
EMBEDDING_MODEL_NAME = SpanAttributes.EMBEDDING_MODEL_NAME
EMBEDDING_TEXT = EmbeddingAttributes.EMBEDDING_TEXT
EMBEDDING_VECTOR = EmbeddingAttributes.EMBEDDING_VECTOR
INPUT_MIME_TYPE = SpanAttributes.INPUT_MIME_TYPE
INPUT_VALUE = SpanAttributes.INPUT_VALUE
LLM_INPUT_MESSAGES = SpanAttributes.LLM_INPUT_MESSAGES
LLM_INVOCATION_PARAMETERS = SpanAttributes.LLM_INVOCATION_PARAMETERS
LLM_MODEL_NAME = SpanAttributes.LLM_MODEL_NAME
LLM_FINISH_REASON = SpanAttributes.LLM_FINISH_REASON
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
LLM_PROMPTS = SpanAttributes.LLM_PROMPTS
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
LLM_PROMPT_TEMPLATE_VERSION = SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION
LLM_TOKEN_COUNT_COMPLETION = SpanAttributes.LLM_TOKEN_COUNT_COMPLETION
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ = SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ
LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE = (
    SpanAttributes.LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE
)
LLM_TOKEN_COUNT_PROMPT = SpanAttributes.LLM_TOKEN_COUNT_PROMPT
LLM_TOKEN_COUNT_TOTAL = SpanAttributes.LLM_TOKEN_COUNT_TOTAL
LLM_TOOLS = SpanAttributes.LLM_TOOLS
MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT

MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON = MessageAttributes.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON
MESSAGE_FUNCTION_CALL_NAME = MessageAttributes.MESSAGE_FUNCTION_CALL_NAME
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_TOOL_CALLS = MessageAttributes.MESSAGE_TOOL_CALLS
MESSAGE_TOOL_CALL_ID = MessageAttributes.MESSAGE_TOOL_CALL_ID
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_SIGNATURE = MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE
MESSAGE_CONTENT_DATA = MessageContentAttributes.MESSAGE_CONTENT_DATA
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
TOOL_CALL_ID = ToolCallAttributes.TOOL_CALL_ID
TOOL_CALL_FUNCTION_ARGUMENTS_JSON = ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
TOOL_CALL_FUNCTION_NAME = ToolCallAttributes.TOOL_CALL_FUNCTION_NAME
TOOL_JSON_SCHEMA = ToolAttributes.TOOL_JSON_SCHEMA
LLM_PROMPT_TEMPLATE = SpanAttributes.LLM_PROMPT_TEMPLATE
LLM_PROMPT_TEMPLATE_VARIABLES = SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES
USER_ID = SpanAttributes.USER_ID
LLM_PROVIDER = SpanAttributes.LLM_PROVIDER
LLM_SYSTEM = SpanAttributes.LLM_SYSTEM
LLM_PROVIDER_ANTHROPIC = OpenInferenceLLMProviderValues.ANTHROPIC.value
LLM_SYSTEM_ANTHROPIC = OpenInferenceLLMSystemValues.ANTHROPIC.value
