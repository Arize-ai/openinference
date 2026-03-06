# ruff: noqa: E501
import json
import random
import string
from typing import Any, Dict, Generator, Optional

import anthropic
import pytest
from anthropic import Anthropic, AsyncAnthropic
from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
from anthropic.resources.beta.messages import Messages as BetaMessages
from anthropic.resources.completions import AsyncCompletions, Completions
from anthropic.resources.messages import AsyncMessages, Messages
from anthropic.types import (
    ImageBlockParam,
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util._importlib_metadata import entry_points
from pydantic import BaseModel
from wrapt import BoundFunctionWrapper

from openinference.instrumentation import OITracer, using_attributes
from openinference.instrumentation.anthropic import AnthropicInstrumentor
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


def remove_all_vcr_request_headers(request: Any) -> Any:
    """
    Removes all request headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_request_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    request.headers.clear()
    return request


def remove_all_vcr_response_headers(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Removes all response headers.

    Example:
    ```
    @pytest.mark.vcr(
        before_record_response=remove_all_vcr_response_headers
    )
    def test_openai() -> None:
        # make request to OpenAI
    """
    response["headers"] = {}
    return response


def _get_tool_use_id(message: Message) -> Optional[str]:
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            return block.id
    return None


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_anthropic_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)
    yield
    AnthropicInstrumentor().uninstrument()


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


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_completions_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="sk-ant-fake")

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" why is the sky blue? respond in five words or less."
        f" {anthropic.AI_PROMPT}"
    )

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
    assert json.loads(output_value) == {
        "completion": " Light scatters blue.",
        "stop": "\n\nHuman:",
        "stop_reason": "stop_sequence",
        "id": "compl_015dfgyiT7JLszAiMbGMtgeG",
        "model": "claude-2.1",
        "type": "completion",
        "log_id": "compl_015dfgyiT7JLszAiMbGMtgeG",
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)

    invocation_params = {"max_tokens_to_sample": 1000, "stream": True}
    assert json.loads(inv_params) == invocation_params
    assert attributes.pop(LLM_OUTPUT_MESSAGES) == " Light scatters blue."
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
        for _ in stream.text_stream:
            pass

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

    msg_out = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
    assert isinstance(msg_out, str)
    assert "paris" in msg_out.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop("llm.token_count.total"), int)

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    raw_inv = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(raw_inv, str)
    assert json.loads(raw_inv) == invocation_params

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
        async for _ in stream.text_stream:
            pass

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

    msg_out = attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}")
    assert isinstance(msg_out, str)
    assert "paris" in msg_out.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON
    assert isinstance(attributes.pop("llm.token_count.total"), int)

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    raw_inv = attributes.pop(LLM_INVOCATION_PARAMETERS)
    assert isinstance(raw_inv, str)
    assert json.loads(raw_inv) == invocation_params

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_anthropic_instrumentation_async_completions_streaming(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="sk-ant-fake")

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" why is the sky blue? respond in five words or less."
        f" {anthropic.AI_PROMPT}"
    )

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
    assert json.loads(output_value) == {
        "completion": " Light scatters blue.",
        "stop": "\n\nHuman:",
        "stop_reason": "stop_sequence",
        "id": "compl_01Ho8r6LNPQ9EVEAh3vpiUnQ",
        "model": "claude-2.1",
        "type": "completion",
        "log_id": "compl_01Ho8r6LNPQ9EVEAh3vpiUnQ",
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-2.1"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)

    invocation_params = {"max_tokens_to_sample": 1000, "stream": True}
    assert json.loads(inv_params) == invocation_params
    assert attributes.pop(LLM_OUTPUT_MESSAGES) == " Light scatters blue."
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_completions(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = Anthropic(api_key="sk-ant-fake")

    invocation_params = {"max_tokens_to_sample": 1000}

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" how does a court case get to the Supreme Court?"
        f" {anthropic.AI_PROMPT}"
    )

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
    assert json.loads(output_value) == {
        "id": "compl_01N6jAWfEZtyE338jUQFx9LC",
        "completion": ' A court case can reach the Supreme Court in a few different ways:\n\n1. Appeal from lower courts. Most cases that reach the Supreme Court are appeals from decisions of federal courts of appeals or state supreme courts. Typically there has to be an important constitutional issue or federal law question for the Supreme Court to accept such an appeal.\n\n2. Original jurisdiction cases. The Supreme Court has original jurisdiction over certain types of cases, meaning they can hear the case directly without it coming from a lower court. These include cases between two or more U.S. states or cases involving ambassadors and other diplomats.\n\n3. Certiorari. This is the process by which the Supreme Court selects most of the cases it hears. Parties to a case petition the Court to review the case, and the Court grants "cert" if four of the nine justices agree to hear it. The Court typically grants certiorari in cases that have broad legal impact or important constitutional questions.\n\n4. Certificate from appeals courts. A federal appeals court can also ask the Supreme Court to take a case by granting a certificate of ascertainability. This happens when the appeals court determines there is a critical question of law that requires the Supreme Court\'s review. \n\nSo in most cases, the Supreme Court exercises discretionary review via petitions for certiorari or certificates from lower courts. Its original jurisdiction over certain types of cases also allows some direct access for parties.',
        "model": "claude-2.1",
        "stop_reason": "stop_sequence",
        "type": "completion",
        "stop": "\n\nHuman:",
        "log_id": "compl_01N6jAWfEZtyE338jUQFx9LC",
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
def test_anthropic_instrumentation_messages(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
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
    assert isinstance(
        msg_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "paris" in msg_content.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert isinstance(
        msg_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
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
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert isinstance(
        msg_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
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
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
async def test_anthropic_instrumentation_async_completions(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_anthropic_instrumentation: Any,
) -> None:
    client = AsyncAnthropic(api_key="sk-ant-fake")

    invocation_params = {"max_tokens_to_sample": 1000}

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" how does a court case get to the Supreme Court?"
        f" {anthropic.AI_PROMPT}"
    )

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
    assert json.loads(output_value) == {
        "id": "compl_01UXLihn1JiHdBhcGahQv7pe",
        "completion": " A court case can reach the Supreme Court in a few different ways:\n\n1. Appeal from lower courts. Most cases that reach the Supreme Court are appeals from decisions at lower federal courts or state supreme courts. Typically, a party who loses at a lower court level can appeal the decision to the next higher court. After the court of appeals, the next stop is the Supreme Court.\n\n2. Original jurisdiction. The Supreme Court has original jurisdiction over certain types of cases, meaning they can hear the case directly without it coming from a lower court. These mainly include cases between two or more states or certain cases involving ambassadors and public ministers.\n\n3. Writ of certiorari. This a process where a party petitions the Supreme Court to hear an appeal from a lower court. The Supreme Court then has discretion on whether or not it wants to hear the case. Each term, there are thousands of petitions for writ of certiorari, but the court only agrees to hear argument in about 100-150 cases per session. \n\n4. Certificate from lower courts or government. Sometimes a circuit court of appeals can certify a legal issue to the Supreme Court before making a final ruling. Government agencies can also refer cases or issues over which there is some uncertainty or disagreement over the correct legal interpretation.\n\nSo in summary, it's usually an appeals process from lower courts, the court's original jurisdiction, or the Supreme Court agreeing to exercise its discretion in hearing an appeal petition on a disputed legal issue. The type and complexity of cases it hears is very selective.",
        "model": "claude-2.1",
        "stop_reason": "stop_sequence",
        "type": "completion",
        "stop": "\n\nHuman:",
        "log_id": "compl_01UXLihn1JiHdBhcGahQv7pe",
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_PROMPTS) == (prompt,)
    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert isinstance(
        msg_content := attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str
    )
    assert "paris" in msg_content.lower()
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert isinstance(attributes.pop(INPUT_VALUE), str)
    assert attributes.pop(INPUT_MIME_TYPE) == JSON
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
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
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == {
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
    }
    assert isinstance(attributes.pop(OUTPUT_MIME_TYPE), str)
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_NAME}")
        == "get_time"
    )
    get_time_input_str = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.1.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    json.loads(get_time_input_str) == {"timezone": "America/New_York"}  # type: ignore
    assert (
        attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    get_weather_input_str = attributes.pop(
        f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_TOOL_CALLS}.0.{TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    assert json.loads(get_weather_input_str) == {"location": "New York, NY"}  # type: ignore
    assert attributes.pop(LLM_TOKEN_COUNT_PROMPT) == 721
    assert attributes.pop(LLM_TOKEN_COUNT_COMPLETION) == 113
    assert attributes.pop(LLM_TOKEN_COUNT_TOTAL) == 834
    output_value = attributes.pop(OUTPUT_VALUE)
    assert isinstance(output_value, str)
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert not attributes


@pytest.mark.vcr(
    record_mode="once",  # allow first recording
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    stream = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=input_messages,
        stream=True,
    )
    events = [event for event in stream]
    assert len(events) > 0
    spans = in_memory_span_exporter.get_finished_spans()
    assert spans[0].name == "messages.create"
    attributes: Dict[str, Any] = dict(spans[0].attributes or dict())
    assert attributes.pop(LLM_MODEL_NAME) == "claude-3-5-sonnet-20240620"
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert isinstance(attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
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
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}").startswith(
        "This image shows the iconic Taj Mahal"
    )
    assert attributes.pop(f"{LLM_TOKEN_COUNT_COMPLETION}") == 296
    assert attributes.pop(f"{LLM_TOKEN_COUNT_PROMPT}") == 78
    assert attributes.pop(f"{LLM_TOKEN_COUNT_TOTAL}") == 374
    assert attributes.pop(f"{OPENINFERENCE_SPAN_KIND}") == "LLM"
    assert not attributes


@pytest.mark.vcr(
    record_mode="once",  # allow first recording
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert attributes.pop(LLM_PROVIDER) == LLM_PROVIDER_ANTHROPIC
    assert attributes.pop(LLM_SYSTEM) == LLM_SYSTEM_ANTHROPIC
    assert isinstance(attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)
    assert attributes.pop(f"{LLM_INPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "user"
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
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}").startswith(
        "This image shows the iconic Taj Mahal"
    )
    assert attributes.pop(f"{LLM_TOKEN_COUNT_COMPLETION}") == 263
    assert attributes.pop(f"{LLM_TOKEN_COUNT_PROMPT}") == 78
    assert attributes.pop(f"{OPENINFERENCE_SPAN_KIND}") == "LLM"
    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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

    prompt = (
        f"{anthropic.HUMAN_PROMPT}"
        f" how does a court case get to the Supreme Court?"
        f" {anthropic.AI_PROMPT}"
    )

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

    for span in spans:
        att = dict(span.attributes or {})
        assert att.get(SESSION_ID, None)
        assert att.get(USER_ID, None)
        assert att.get(METADATA, None)
        assert att.get(TAG_TAGS, None)
        assert att.get(LLM_PROMPT_TEMPLATE, None)
        assert att.get(LLM_PROMPT_TEMPLATE_VERSION, None)
        assert att.get(LLM_PROMPT_TEMPLATE_VARIABLES, None)


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert json.loads(output_value) == {
        "id": "msg_017pC17fmFPUhGb5UENdPKqG",
        "container": None,
        "content": [
            {
                "citations": None,
                "text": "{\"city\":\"Paris\",\"country\":\"France\"}",
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
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
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert json.loads(output_value) == {
        "id": "msg_01UxkoYKRxHPYTUYkGic5teK",
        "container": None,
        "content": [
            {
                "citations": None,
                "text": "{\"city\":\"Paris\",\"country\":\"France\"}",
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
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
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert not attributes


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert json.loads(output_value) == {
        "id": "msg_01CiA3YpvhgJbxvaoofq8Pri",
        "container": None,
        "content": [
            {
                "citations": None,
                "text": "{\"city\":\"Paris\",\"country\":\"France\"}",
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
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
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert json.loads(output_value) == {
        "id": "msg_01YLp4hqTXinnBRQ6MMipuy9",
        "container": None,
        "content": [
            {
                "citations": None,
                "text": "{\"city\":\"Paris\",\"country\":\"France\"}",
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert isinstance(attributes.pop(LLM_MODEL_NAME), str)
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
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert not attributes


def test_anthropic_uninstrumentation(
    tracer_provider: TracerProvider,
) -> None:
    AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

    assert isinstance(Completions.create, BoundFunctionWrapper)
    assert isinstance(AsyncCompletions.create, BoundFunctionWrapper)

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

    assert not isinstance(Completions.create, BoundFunctionWrapper)
    assert not isinstance(AsyncCompletions.create, BoundFunctionWrapper)

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


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

    assert not attributes


@pytest.mark.asyncio
@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=remove_all_vcr_request_headers,
    before_record_response=remove_all_vcr_response_headers,
)
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
    assert json.loads(output_value) == {
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
    }
    assert attributes.pop(OUTPUT_MIME_TYPE) == JSON

    assert attributes.pop(LLM_MODEL_NAME) == "claude-sonnet-4-6"
    assert isinstance(inv_params := attributes.pop(LLM_INVOCATION_PARAMETERS), str)
    assert json.loads(inv_params) == invocation_params

    assert attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_ROLE}") == "assistant"
    assert isinstance(attributes.pop(f"{LLM_OUTPUT_MESSAGES}.0.{MESSAGE_CONTENT}"), str)

    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_PROMPT), int)
    assert isinstance(attributes.pop(LLM_TOKEN_COUNT_COMPLETION), int)

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
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
METADATA = SpanAttributes.METADATA
OPENINFERENCE_SPAN_KIND = SpanAttributes.OPENINFERENCE_SPAN_KIND
OUTPUT_MIME_TYPE = SpanAttributes.OUTPUT_MIME_TYPE
OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE
RETRIEVAL_DOCUMENTS = SpanAttributes.RETRIEVAL_DOCUMENTS
SESSION_ID = SpanAttributes.SESSION_ID
TAG_TAGS = SpanAttributes.TAG_TAGS
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
