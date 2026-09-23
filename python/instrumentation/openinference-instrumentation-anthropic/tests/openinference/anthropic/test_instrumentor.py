# ruff: noqa: E501
import asyncio
import json
import random
import string
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    ClassVar,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
)

import anthropic
import httpx2
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
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from pydantic import BaseModel, field_validator
from wrapt import BoundFunctionWrapper, FunctionWrapper

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.anthropic import (
    AnthropicInstrumentor,
    _get_anthropic_version,
)
from openinference.instrumentation.anthropic._stream import _MessageExtractor
from openinference.instrumentation.anthropic._wrappers import (
    _get_llm_input_messages,
    _get_llm_token_counts,
    _get_output_messages,
    _Params,
    _PrepareRequestDataWrapper,
    _TransformWrapper,
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


def _mock_anthropic_client(handler: Callable[[Any], Any]) -> Anthropic:
    """Build an ``Anthropic`` client whose HTTP transport is mocked by ``handler``."""
    transport = httpx2.MockTransport(handler)
    return Anthropic(api_key="sk-ant-fake", http_client=httpx2.Client(transport=transport))


def _mock_async_anthropic_client(handler: Callable[[Any], Any]) -> AsyncAnthropic:
    """Build an ``AsyncAnthropic`` client whose HTTP transport is mocked by ``handler``."""
    transport = httpx2.MockTransport(handler)
    return AsyncAnthropic(
        api_key="sk-ant-fake", http_client=httpx2.AsyncClient(transport=transport)
    )


_STREAM_KWARGS: Dict[str, Any] = {
    "model": "claude-sonnet-4-6",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "hello"}],
}


def _bad_request_handler(request: Any) -> Any:
    """A non-retryable error response, so the client does not back off before raising."""
    return httpx2.Response(
        status_code=400,
        json={"type": "error", "error": {"type": "invalid_request_error", "message": "nope"}},
    )


_MESSAGE_JSON: Dict[str, Any] = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


def _message_handler(request: Any) -> Any:
    return httpx2.Response(status_code=200, json=_MESSAGE_JSON)


def _unread_message_handler(request: Any) -> Any:
    """A response whose body is streamed from the transport, so it stays unread until consumed."""
    return httpx2.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        content=iter([json.dumps(_MESSAGE_JSON).encode()]),
    )


def _async_unread_message_handler(request: Any) -> Any:
    async def content() -> Any:
        yield json.dumps(_MESSAGE_JSON).encode()

    return httpx2.Response(
        status_code=200, headers={"content-type": "application/json"}, content=content()
    )


def _event_stream_body(error: bool = False) -> bytes:
    """_MESSAGE_JSON as server-sent events, or ending in an error event instead of message_stop."""
    events: List[Dict[str, Any]] = [
        {"type": "message_start", "message": {**_MESSAGE_JSON, "content": []}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    if error:
        events[-1] = {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        }
    body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
    return body.encode()


def _event_stream_handler(request: Any) -> Any:
    return httpx2.Response(
        status_code=200, headers={"content-type": "text/event-stream"}, content=_event_stream_body()
    )


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

    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-sonnet-4-6"
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

    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-sonnet-4-6"
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
    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-sonnet-4-6"
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
    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-sonnet-4-6"
    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_messages_model_fallback(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """Covers provider-side classifier/fallback routing: the response is served
    by a different model than the one requested (e.g. Opus 5 requested, routed
    to Opus 4.8). llm.request.model_name and llm.response.model_name must be
    captured as distinct values, and llm.model_name must reflect the model
    that actually served the response.
    """
    client = Anthropic(api_key="sk-ant-fake")
    input_message = "What's the capital of France?"

    invocation_params = {"max_tokens": 1024}

    client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-opus-5",
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
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == "The capital of France is **Paris**."
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 14
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 11
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 25

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(output_value, {"model": "claude-opus-4-8"})
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-opus-5"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-opus-4-8"
    assert attributes.pop(LLM_MODEL_NAME) == "claude-opus-4-8"
    assert not attributes


@pytest.mark.vcr
def test_anthropic_instrumentation_messages_streaming_model_fallback(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """Streaming counterpart of test_anthropic_instrumentation_messages_model_fallback.
    Also guards against the pre-fix regression where the streaming path never
    read the response model at all, leaving llm.model_name stuck at the
    requested model.
    """
    client = Anthropic(api_key="sk-ant-fake")
    input_message = "What's the capital of France?"

    invocation_params = {"max_tokens": 1024, "stream": True}

    stream = client.messages.create(
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": input_message,
            }
        ],
        model="claude-opus-5",
        stream=True,
    )

    for _ in stream:
        pass

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
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENTS}.0.{MESSAGE_CONTENT_TEXT}")
        == "The capital of France is **Paris**."
    )
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 14
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 11
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 25

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert_output_value_contains(output_value, {"model": "claude-opus-4-8"})
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_FINISH_REASON, None) == "end_turn"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-opus-5"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-opus-4-8"
    assert attributes.pop(LLM_MODEL_NAME) == "claude-opus-4-8"
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
    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-sonnet-4-6"
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

    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-sonnet-4-6"
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
    assert isinstance(attributes.pop(LLM_REQUEST_MODEL_NAME), str)
    assert isinstance(attributes.pop(LLM_RESPONSE_MODEL_NAME), str)
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
    assert isinstance(attributes.pop(LLM_REQUEST_MODEL_NAME), str)
    assert isinstance(attributes.pop(LLM_RESPONSE_MODEL_NAME), str)
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
    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-3-5-sonnet-20240620"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-3-5-sonnet-20240620"
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
    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-3-5-sonnet-20240620"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-3-5-sonnet-20240620"
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

    attributes = dict(spans[0].attributes or {})

    assert (
        attributes.get(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert (
        attributes.get(
            f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
        )
        == '{"location": "San Francisco, CA", "unit": "fahrenheit"}'
    )
    assert attributes.get(f"{LLM_INPUT_MESSAGES}.1.{MESSAGE_ROLE}") == "assistant"

    assert (
        attributes.get(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_CONTENT}")
        == '{"weather": "sunny", "temperature": "75"}'
    )
    assert attributes.get(f"{LLM_INPUT_MESSAGES}.2.{MESSAGE_ROLE}") == "user"


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
            messages=[
                {"role": "user", "content": "How does a court case get to the Supreme Court?"}
            ],
            max_tokens=1000,
        )

    spans = in_memory_span_exporter.get_finished_spans()

    for span in spans:
        att = dict(span.attributes or {})
        assert att.get(SESSION_ID, None)
        assert att.get(USER_ID, None)
        assert att.get(METADATA, None)
        assert att.get(TAG_TAGS, None)
        assert att.get(LLM_PROMPT_TEMPLATE, None)
        assert att.get(LLM_PROMPT_TEMPLATE_VERSION, None)
        assert att.get(LLM_PROMPT_TEMPLATE_VARIABLES, None)


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
    assert att1.get(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ) is None
    # second request doesn't write cache
    assert att2.get(LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE) is None


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

    assert isinstance(attributes.pop(LLM_REQUEST_MODEL_NAME), str)
    assert isinstance(attributes.pop(LLM_RESPONSE_MODEL_NAME), str)
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

    assert isinstance(attributes.pop(LLM_REQUEST_MODEL_NAME), str)
    assert isinstance(attributes.pop(LLM_RESPONSE_MODEL_NAME), str)
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

    assert isinstance(attributes.pop(LLM_REQUEST_MODEL_NAME), str)
    assert isinstance(attributes.pop(LLM_RESPONSE_MODEL_NAME), str)
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

    assert isinstance(attributes.pop(LLM_REQUEST_MODEL_NAME), str)
    assert isinstance(attributes.pop(LLM_RESPONSE_MODEL_NAME), str)
    assert not attributes


def test_anthropic_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

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


def test_request_body_preparation_is_instrumented_and_restored(
    tracer_provider: TracerProvider,
) -> None:
    """
    The private request body preparation functions are patched to enrich the recorded
    invocation parameters. anthropic 1.8.0 renamed them and moved their call site, so this
    fails if a future version moves them again, which instrument() only logs a warning for.
    """
    anthropic_version = _get_anthropic_version()
    assert anthropic_version is not None, anthropic.__version__
    if anthropic_version >= (1, 8, 0):
        import anthropic._base_client as module

        sync_name, async_name = "prepare_request_data", "async_prepare_request_data"
    else:
        import anthropic._utils._transform as module  # type: ignore[no-redef]

        sync_name, async_name = "transform", "async_transform"
    original = getattr(module, sync_name)
    async_original = getattr(module, async_name)

    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    try:
        assert isinstance(getattr(module, sync_name), FunctionWrapper)
        assert isinstance(getattr(module, async_name), FunctionWrapper)
    finally:
        # the instrumentor is a singleton, so a failure here would otherwise leave every
        # later test instrumented against this test's tracer provider
        AnthropicInstrumentor().uninstrument()

    assert getattr(module, sync_name) is original
    assert getattr(module, async_name) is async_original


@pytest.mark.parametrize(
    "wrapper,location,expected",
    [
        pytest.param(_TransformWrapper(), {}, {"max_tokens": 256, "stream": True}, id="transform"),
        pytest.param(
            _PrepareRequestDataWrapper(),
            {"location": "body"},
            {"max_tokens": 256, "stream": True},
            id="prepare_request_data-body",
        ),
        pytest.param(
            _PrepareRequestDataWrapper(),
            {"location": "query"},
            {"max_tokens": 256},
            id="prepare_request_data-query",
        ),
    ],
)
def test_only_request_bodies_are_recorded_as_invocation_parameters(
    wrapper: Callable[..., Any],
    location: Dict[str, str],
    expected: Dict[str, Any],
) -> None:
    """
    anthropic<1.8.0 prepares request bodies only. anthropic>=1.8.0 prepares request bodies and
    query parameters with the same function, telling them apart with a ``location`` keyword.
    Only bodies carry invocation parameters.
    """
    prepared = {"stream": True}

    def prepare(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return prepared

    params = _Params({"max_tokens": 256})
    with params:
        assert wrapper(prepare, None, (prepared,), location) is prepared
    assert dict(params) == expected


@pytest.mark.parametrize("beta", [False, True], ids=["messages", "beta_messages"])
def test_failed_streaming_request_is_recorded(
    beta: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    The streaming helpers only make the request when the manager is entered, so a request
    that fails there must still end the span.
    """
    client = _mock_anthropic_client(_bad_request_handler)
    messages: Any = client.beta.messages if beta else client.messages

    with pytest.raises(anthropic.BadRequestError):
        with messages.stream(**_STREAM_KWARGS):
            pass

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.events
    attributes = dict(span.attributes or {})
    assert json.loads(str(attributes[LLM_INVOCATION_PARAMETERS])) == {
        "max_tokens": 1000,
        "stream": True,
    }


@pytest.mark.parametrize("beta", [False, True], ids=["messages", "beta_messages"])
async def test_failed_async_streaming_request_is_recorded(
    beta: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_async_anthropic_client(_bad_request_handler)
    messages: Any = client.beta.messages if beta else client.messages

    with pytest.raises(anthropic.BadRequestError):
        async with messages.stream(**_STREAM_KWARGS):
            pass

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.events
    attributes = dict(span.attributes or {})
    assert json.loads(str(attributes[LLM_INVOCATION_PARAMETERS])) == {
        "max_tokens": 1000,
        "stream": True,
    }


@pytest.mark.parametrize("beta", [False, True], ids=["messages", "beta_messages"])
async def test_cancelled_async_streaming_request_is_recorded(
    beta: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    A cancelled task, e.g. under asyncio.timeout(), raises CancelledError, which is a
    BaseException rather than an Exception, and must still end the span.
    """

    def cancelled_handler(request: Any) -> Any:
        raise asyncio.CancelledError

    client = _mock_async_anthropic_client(cancelled_handler)
    messages: Any = client.beta.messages if beta else client.messages

    with pytest.raises(asyncio.CancelledError):
        async with messages.stream(**_STREAM_KWARGS):
            pass

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.events


@pytest.mark.parametrize("beta", [False, True], ids=["messages", "beta_messages"])
def test_raw_response_is_recorded(
    beta: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    with_raw_response returns the HTTP response instead of the message. Its body has already been
    read, so the message is recorded, and the caller still gets the response it asked for.
    """
    client = _mock_anthropic_client(_message_handler)
    messages: Any = client.beta.messages if beta else client.messages

    response = messages.with_raw_response.create(**_STREAM_KWARGS)

    assert isinstance(response, anthropic.APIResponse)
    assert response.parse().content[0].text == "hi"
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.OK
    attributes = dict(span.attributes or {})
    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert json.loads(str(attributes[OUTPUT_VALUE]))["content"][0]["text"] == "hi"
    assert attributes[LLM_TOKEN_COUNT_PROMPT] == 3


@pytest.mark.parametrize("beta", [False, True], ids=["messages", "beta_messages"])
async def test_async_raw_response_is_recorded(
    beta: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_async_anthropic_client(_message_handler)
    messages: Any = client.beta.messages if beta else client.messages

    response = await messages.with_raw_response.create(**_STREAM_KWARGS)

    assert isinstance(response, anthropic.AsyncAPIResponse)
    assert (await response.parse()).content[0].text == "hi"
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.OK
    attributes = dict(span.attributes or {})
    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    assert json.loads(str(attributes[OUTPUT_VALUE]))["content"][0]["text"] == "hi"
    assert attributes[LLM_TOKEN_COUNT_PROMPT] == 3


class _City(BaseModel):
    city: str
    validations: ClassVar[int] = 0

    @field_validator("city")
    @classmethod
    def count_validations(cls, city: str) -> str:
        cls.validations += 1
        return city


def _parsed_message_handler(request: Any) -> Any:
    content = [{"type": "text", "text": json.dumps({"city": "Paris"})}]
    return httpx2.Response(status_code=200, json={**_MESSAGE_JSON, "content": content})


def test_raw_parse_response_is_left_to_the_caller(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    parse() validates the response against the caller's output_format, which is caller code
    that must run only when the caller parses the raw response itself.
    """
    _City.validations = 0
    client = _mock_anthropic_client(_parsed_message_handler)

    response = client.beta.messages.with_raw_response.parse(**_STREAM_KWARGS, output_format=_City)

    assert _City.validations == 0
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.OK
    assert response.parse().parsed_output == _City(city="Paris")


@pytest.mark.parametrize("beta", [False, True], ids=["messages", "beta_messages"])
def test_raw_response_with_middleware_post_parser_is_left_to_the_caller(
    beta: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    Middleware can attach a post_parser to any request, which is caller code that must run only
    when the caller parses the raw response itself.
    """
    post_parsed: List[Any] = []

    def post_parser(message: Any) -> Any:
        post_parsed.append(message)
        return message

    def middleware(request: Any, call_next: Callable[[Any], Any]) -> Any:
        request = request.copy()
        request.options.post_parser = post_parser
        return call_next(request)

    client = Anthropic(
        api_key="sk-ant-fake",
        middleware=[middleware],
        http_client=httpx2.Client(transport=httpx2.MockTransport(_message_handler)),
    )
    messages: Any = client.beta.messages if beta else client.messages

    response = messages.with_raw_response.create(**_STREAM_KWARGS)

    assert not post_parsed
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.OK
    assert response.parse().content[0].text == "hi"
    assert len(post_parsed) == 1


async def test_async_raw_parse_response_is_left_to_the_caller(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    _City.validations = 0
    client = _mock_async_anthropic_client(_parsed_message_handler)

    response = await client.beta.messages.with_raw_response.parse(
        **_STREAM_KWARGS, output_format=_City
    )

    assert _City.validations == 0
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.OK
    assert (await response.parse()).parsed_output == _City(city="Paris")


@pytest.mark.parametrize("buffered", [False, True], ids=["streamed", "buffered"])
def test_raw_event_stream_status_is_left_unset(
    buffered: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    A raw streaming response is returned before its events are read, and a stream can still end
    in an error event, so the span must not be recorded as successful. A transport can buffer the
    body, e.g. a recorded cassette, which closes the response although its events are unread.
    """
    body = _event_stream_body(error=True)

    def handler(request: Any) -> Any:
        return httpx2.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=body if buffered else iter([body]),
        )

    client = _mock_anthropic_client(handler)

    response = client.messages.with_raw_response.create(**_STREAM_KWARGS, stream=True)
    (span,) = in_memory_span_exporter.get_finished_spans()
    with pytest.raises(anthropic.APIStatusError):
        for _ in response.parse():
            pass

    assert span.status.status_code == trace_api.StatusCode.UNSET


@pytest.mark.parametrize("beta", [False, True], ids=["messages", "beta_messages"])
def test_streaming_response_body_is_left_to_the_caller(
    beta: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    with_streaming_response leaves the body unread for the caller, so it is not parsed.
    """
    client = _mock_anthropic_client(_unread_message_handler)
    messages: Any = client.beta.messages if beta else client.messages

    with messages.with_streaming_response.create(**_STREAM_KWARGS) as response:
        assert not response.is_closed
        (span,) = in_memory_span_exporter.get_finished_spans()
        assert response.parse().content[0].text == "hi"

    assert span.status.status_code == trace_api.StatusCode.UNSET
    assert OUTPUT_VALUE not in (span.attributes or {})


@pytest.mark.parametrize("beta", [False, True], ids=["messages", "beta_messages"])
async def test_async_streaming_response_body_is_left_to_the_caller(
    beta: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_async_anthropic_client(_async_unread_message_handler)
    messages: Any = client.beta.messages if beta else client.messages

    async with messages.with_streaming_response.create(**_STREAM_KWARGS) as response:
        assert not response.is_closed
        (span,) = in_memory_span_exporter.get_finished_spans()
        assert (await response.parse()).content[0].text == "hi"

    assert span.status.status_code == trace_api.StatusCode.UNSET
    assert OUTPUT_VALUE not in (span.attributes or {})


@pytest.mark.parametrize(
    "extra_body",
    [
        pytest.param({"self": "value"}, id="self"),
        pytest.param({1: "value"}, id="non-string"),
    ],
)
def test_any_request_body_key_is_recorded_without_raising(
    extra_body: Dict[Any, Any],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    anthropic>=1.8.0 merges extra_body into the request body before it is prepared, so the
    prepared body recorded as invocation parameters can have any keys.
    """
    sent: List[Dict[str, Any]] = []

    def handler(request: Any) -> Any:
        sent.append(json.loads(request.content))
        return _message_handler(request)

    client = _mock_anthropic_client(handler)

    client.messages.create(**_STREAM_KWARGS, extra_body=extra_body)

    ((key, value),) = extra_body.items()
    assert sent[0][str(key)] == value
    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.OK
    anthropic_version = _get_anthropic_version()
    assert anthropic_version is not None, anthropic.__version__
    if anthropic_version >= (1, 8, 0):
        # earlier versions prepare the body before extra_body is merged into it
        attributes = dict(span.attributes or {})
        assert json.loads(str(attributes[LLM_INVOCATION_PARAMETERS]))[str(key)] == value


@pytest.mark.parametrize("exhaust", [True, False], ids=["exhausted", "left_early"])
def test_streaming_create_as_context_manager_is_recorded(
    exhaust: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    The SDK stream's context manager returns the SDK stream itself, which bypassed the
    instrumented iteration, and leaving the context early never finished the span.
    """
    client = _mock_anthropic_client(_event_stream_handler)

    with client.messages.create(**_STREAM_KWARGS, stream=True) as stream:
        for _ in stream:
            if not exhaust:
                break

    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    if exhaust:
        assert span.status.status_code == trace_api.StatusCode.OK
        assert json.loads(str(attributes[OUTPUT_VALUE]))["content"][0]["text"] == "hi"


@pytest.mark.parametrize("exhaust", [True, False], ids=["exhausted", "left_early"])
async def test_async_streaming_create_as_context_manager_is_recorded(
    exhaust: bool,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_async_anthropic_client(_event_stream_handler)

    async with await client.messages.create(**_STREAM_KWARGS, stream=True) as stream:
        async for _ in stream:
            if not exhaust:
                break

    (span,) = in_memory_span_exporter.get_finished_spans()
    attributes = dict(span.attributes or {})
    assert attributes[f"{LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"] == "assistant"
    if exhaust:
        assert span.status.status_code == trace_api.StatusCode.OK
        assert json.loads(str(attributes[OUTPUT_VALUE]))["content"][0]["text"] == "hi"


def test_exception_leaving_streaming_create_context_is_recorded(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_anthropic_client(_event_stream_handler)

    with pytest.raises(RuntimeError):
        with client.messages.create(**_STREAM_KWARGS, stream=True) as stream:
            for _ in stream:
                raise RuntimeError("stop")

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.events


def test_exception_closing_streaming_create_context_is_recorded(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    Leaving the context closes the response, which can itself fail after the body of the
    context completed normally.
    """

    class FailingToCloseStream(httpx2.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            yield _event_stream_body()

        def close(self) -> None:
            raise OSError("close failed")

    def handler(request: Any) -> Any:
        return httpx2.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            stream=FailingToCloseStream(),
        )

    client = _mock_anthropic_client(handler)

    with pytest.raises(OSError):
        with client.messages.create(**_STREAM_KWARGS, stream=True) as stream:
            next(iter(stream))

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.events


def test_closing_generator_holding_streaming_create_context_is_not_an_error(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    Closing a generator that holds the context raises GeneratorExit through it, which leaves the
    stream early rather than failing the request.
    """
    client = _mock_anthropic_client(_event_stream_handler)

    def events() -> Generator[Any, None, None]:
        with client.messages.create(**_STREAM_KWARGS, stream=True) as stream:
            yield from stream

    generator = events()
    next(generator)
    generator.close()

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.UNSET
    assert not span.events


async def test_closing_async_generator_holding_streaming_create_context_is_not_an_error(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = _mock_async_anthropic_client(_event_stream_handler)

    async def events() -> AsyncGenerator[Any, None]:
        async with await client.messages.create(**_STREAM_KWARGS, stream=True) as stream:
            async for event in stream:
                yield event

    generator = events()
    await generator.__anext__()
    await generator.aclose()

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.UNSET
    assert not span.events


async def test_cancellation_leaving_async_streaming_create_context_is_recorded(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    A task cancelled while awaiting the next event raises CancelledError, which is a
    BaseException that iteration does not catch, so the context has to record it.
    """
    first_event_read = asyncio.Event()

    def handler(request: Any) -> Any:
        async def content() -> Any:
            yield _event_stream_body().split(b"\n\n", 1)[0] + b"\n\n"
            await asyncio.Event().wait()  # the next event never arrives

        return httpx2.Response(
            status_code=200, headers={"content-type": "text/event-stream"}, content=content()
        )

    client = _mock_async_anthropic_client(handler)

    async def consume() -> None:
        async with await client.messages.create(**_STREAM_KWARGS, stream=True) as stream:
            async for _ in stream:
                first_event_read.set()

    task = asyncio.create_task(consume())
    await first_event_read.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.events


@pytest.mark.parametrize(
    "method,kwargs",
    [
        pytest.param("create", {}, id="create"),
        pytest.param("create", {"stream": True}, id="create_stream"),
        pytest.param("parse", {}, id="parse"),
    ],
)
async def test_cancelled_async_request_is_recorded(
    method: str,
    kwargs: Dict[str, Any],
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    """
    A cancelled task, e.g. under asyncio.timeout(), raises CancelledError, which is a
    BaseException rather than an Exception, and must still end the span.
    """

    def cancelled_handler(request: Any) -> Any:
        raise asyncio.CancelledError

    client = _mock_async_anthropic_client(cancelled_handler)

    with pytest.raises(asyncio.CancelledError):
        await getattr(client.messages, method)(**_STREAM_KWARGS, **kwargs)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.events


def test_interrupted_request_is_recorded(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    def interrupted_handler(request: Any) -> Any:
        raise KeyboardInterrupt

    client = _mock_anthropic_client(interrupted_handler)

    with pytest.raises(KeyboardInterrupt):
        client.messages.create(**_STREAM_KWARGS)

    (span,) = in_memory_span_exporter.get_finished_spans()
    assert span.status.status_code == trace_api.StatusCode.ERROR
    assert span.events


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

    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-sonnet-4-6"
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

    assert attributes.pop(LLM_REQUEST_MODEL_NAME) == "claude-sonnet-4-6"
    assert attributes.pop(LLM_RESPONSE_MODEL_NAME) == "claude-sonnet-4-6"
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
    sse_events = [
        b"event: message_start\ndata: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": usage,
                },
            }
        ).encode()
        + b"\n\n",
        b"event: content_block_start\ndata: "
        + json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ).encode()
        + b"\n\n",
        b"event: content_block_delta\ndata: "
        + json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            }
        ).encode()
        + b"\n\n",
        b"event: content_block_stop\ndata: "
        + json.dumps({"type": "content_block_stop", "index": 0}).encode()
        + b"\n\n",
        b"event: message_delta\ndata: "
        + json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": usage,
            }
        ).encode()
        + b"\n\n",
        b"event: message_stop\ndata: " + json.dumps({"type": "message_stop"}).encode() + b"\n\n",
    ]
    kwargs: Dict[str, Any] = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "hello"}],
    }

    def json_handler(request: Any) -> Any:
        return httpx2.Response(
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

    def sse_handler(request: Any) -> Any:
        return httpx2.Response(status_code=200, content=b"".join(sse_events))

    _mock_anthropic_client(json_handler).messages.create(**kwargs)

    for _ in _mock_anthropic_client(sse_handler).messages.create(stream=True, **kwargs):
        pass

    with _mock_anthropic_client(sse_handler).messages.stream(**kwargs) as stream:
        for _ in stream:
            pass

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 3
    expected = {
        LLM_TOKEN_COUNT_PROMPT: 2255,
        LLM_TOKEN_COUNT_COMPLETION: 5,
        LLM_TOKEN_COUNT_TOTAL: 2260,
        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ: 512,
        LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE: 1733,
    }
    for span in spans:
        attributes = dict(span.attributes or {})
        assert {k: attributes.get(k) for k in expected} == expected

    # message_content.id must never be emitted for thinking/redacted_thinking blocks
    assert not any(key.endswith("message_content.id") for key in attributes)


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
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    def handler(request: Any) -> Any:
        return httpx2.Response(
            status_code=200,
            json={
                "id": "msg_test123",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = _mock_anthropic_client(handler)
    client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": "hello"}],
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(spans[0].attributes or {})
    assert attributes.get(LLM_FINISH_REASON) == stop_reason


@pytest.mark.parametrize(
    "stop_reason",
    ["end_turn", "max_tokens", "tool_use"],
)
def test_finish_reason_values_messages_create_streaming(
    stop_reason: str,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    sse_events = [
        b"event: message_start\ndata: "
        + json.dumps(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-sonnet-4-6",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                },
            }
        ).encode()
        + b"\n\n",
        b"event: content_block_start\ndata: "
        + json.dumps(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            }
        ).encode()
        + b"\n\n",
        b"event: content_block_delta\ndata: "
        + json.dumps(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            }
        ).encode()
        + b"\n\n",
        b"event: content_block_stop\ndata: "
        + json.dumps({"type": "content_block_stop", "index": 0}).encode()
        + b"\n\n",
        b"event: message_delta\ndata: "
        + json.dumps(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": 5},
            }
        ).encode()
        + b"\n\n",
        b"event: message_stop\ndata: " + json.dumps({"type": "message_stop"}).encode() + b"\n\n",
    ]

    def handler(request: Any) -> Any:
        return httpx2.Response(status_code=200, content=b"".join(sse_events))

    client = _mock_anthropic_client(handler)
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
    assert attributes.get(LLM_FINISH_REASON) == stop_reason


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
LLM_REQUEST_MODEL_NAME = SpanAttributes.LLM_REQUEST_MODEL_NAME
LLM_RESPONSE_MODEL_NAME = SpanAttributes.LLM_RESPONSE_MODEL_NAME
LLM_FINISH_REASON = SpanAttributes.LLM_FINISH_REASON
LLM_OUTPUT_MESSAGES = SpanAttributes.LLM_OUTPUT_MESSAGES
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
