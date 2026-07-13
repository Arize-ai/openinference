import json
from typing import Any, AsyncIterator, Awaitable, Dict, Generator, Iterator, List, Mapping, cast

import litellm.anthropic_interface as litellm_anthropic
import pytest
from opentelemetry import context as context_api
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import (
    OITracer,
    TraceConfig,
    safe_json_dumps,
    using_attributes,
)
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    OpenInferenceLLMSystemValues,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

REDACTED_VALUE = "__REDACTED__"

MODEL = "anthropic/claude-3-haiku-20240307"


def _mock_anthropic_response(text: str = "Hello from Claude") -> Dict[str, Any]:
    return {
        "content": [{"text": text, "type": "text"}],
        "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
        "model": "claude-3-haiku-20240307",
        "role": "assistant",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "type": "message",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _mock_tool_use_response() -> Dict[str, Any]:
    return {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "get_weather",
                "input": {"city": "SF"},
            }
        ],
        "id": "msg_tool",
        "model": "claude-3-haiku-20240307",
        "role": "assistant",
        "stop_reason": "tool_use",
        "type": "message",
        "usage": {"input_tokens": 15, "output_tokens": 25},
    }


def _streaming_events(text: str = "Hi") -> List[Dict[str, Any]]:
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "claude-3-haiku-20240307",
                "content": [],
                "usage": {"input_tokens": 8, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": text},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 4},
        },
        {"type": "message_stop"},
    ]


@pytest.fixture()
def setup_litellm_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield


@pytest.mark.parametrize(
    "call_path",
    ["anthropic.create", "anthropic.messages.create"],
)
def test_create(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    call_path: str,
) -> None:
    in_memory_span_exporter.clear()
    messages = [{"role": "user", "content": "Hello"}]
    system = "You are helpful."

    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]

    def mock_create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return _mock_anthropic_response("Beijing")

    LiteLLMInstrumentor.original_anthropic_funcs["create"] = mock_create
    try:
        create_fn = (
            litellm_anthropic.create
            if call_path == "anthropic.create"
            else litellm_anthropic.messages.create
        )
        create_fn(
            model=MODEL,
            messages=messages,
            max_tokens=64,
            system=system,
        )
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "messages.create"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == (
        OpenInferenceSpanKindValues.LLM.value
    )
    assert attributes.pop(SpanAttributes.LLM_SYSTEM) == OpenInferenceLLMSystemValues.ANTHROPIC.value
    # Response may report the bare model id (without provider prefix)
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) in {
        MODEL,
        "claude-3-haiku-20240307",
    }
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "anthropic"
    assert attributes.pop(
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}"
    ) == ("system")
    assert (
        attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENT}")
        == system
    )
    assert attributes.pop(
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_ROLE}"
    ) == ("user")
    assert (
        attributes.pop(f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}")
        == "Hello"
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": messages, "system": system}
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.JSON.value
    assert json.loads(str(attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS))) == {
        "model": MODEL,
        "max_tokens": 64,
    }
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Beijing"
    assert attributes.pop(SpanAttributes.OUTPUT_MIME_TYPE) == OpenInferenceMimeTypeValues.TEXT.value
    assert (
        attributes.pop(f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_ROLE}")
        == "assistant"
    )
    assert (
        attributes.pop(
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0."
            f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"
        )
        == "text"
    )
    assert (
        attributes.pop(
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.0."
            f"{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"
        )
        == "Beijing"
    )
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert not attributes
    assert span.status.status_code == StatusCode.OK


def test_create_with_positional_arguments(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]
    LiteLLMInstrumentor.original_anthropic_funcs["create"] = (
        lambda *args, **kwargs: _mock_anthropic_response()
    )
    messages = [{"role": "user", "content": "Hello"}]
    try:
        litellm_anthropic.create(64, messages, MODEL)
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    attributes = dict(
        cast(
            Mapping[str, AttributeValue], in_memory_span_exporter.get_finished_spans()[0].attributes
        )
    )
    assert attributes[SpanAttributes.LLM_PROVIDER] == "anthropic"
    assert attributes[SpanAttributes.INPUT_VALUE] == safe_json_dumps({"messages": messages})
    assert json.loads(str(attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS])) == {
        "max_tokens": 64,
        "model": MODEL,
    }


@pytest.mark.asyncio
async def test_acreate(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    original = LiteLLMInstrumentor.original_anthropic_funcs["acreate"]

    async def mock_acreate(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return _mock_anthropic_response("Async hello")

    LiteLLMInstrumentor.original_anthropic_funcs["acreate"] = mock_acreate
    try:
        await litellm_anthropic.acreate(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=32,
        )
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["acreate"] = original

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "messages.create"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Async hello"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert span.status.status_code == StatusCode.OK


def test_create_with_tools(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        }
    ]
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]

    def mock_create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return _mock_tool_use_response()

    LiteLLMInstrumentor.original_anthropic_funcs["create"] = mock_create
    try:
        litellm_anthropic.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Weather in SF?"}],
            max_tokens=64,
            tools=tools,
        )
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    assert attributes.get(
        f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}"
    ) == safe_json_dumps(tools[0])
    assert (
        attributes.get(
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
            f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
        )
        == "get_weather"
    )
    assert (
        attributes.get(
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
            f"{ToolCallAttributes.TOOL_CALL_ID}"
        )
        == "toolu_01"
    )


def test_create_context_attributes(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]
    LiteLLMInstrumentor.original_anthropic_funcs["create"] = (
        lambda *args, **kwargs: _mock_anthropic_response()
    )
    try:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            litellm_anthropic.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=16,
            )
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    assert attributes.get(SpanAttributes.SESSION_ID) == session_id
    assert attributes.get(SpanAttributes.USER_ID) == user_id
    assert json.loads(str(attributes.get(SpanAttributes.METADATA))) == metadata
    assert list(cast(Any, attributes.get(SpanAttributes.TAG_TAGS))) == tags
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE) == prompt_template
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION) == prompt_template_version
    assert attributes.get(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES) == json.dumps(
        prompt_template_variables
    )


def test_create_suppress_instrumentation(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]
    LiteLLMInstrumentor.original_anthropic_funcs["create"] = (
        lambda *args, **kwargs: _mock_anthropic_response()
    )
    token = context_api.attach(context_api.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        litellm_anthropic.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=16,
        )
    finally:
        context_api.detach(token)
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    assert len(in_memory_span_exporter.get_finished_spans()) == 0


def test_create_error_status(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]

    def mock_create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise RuntimeError("boom")

    LiteLLMInstrumentor.original_anthropic_funcs["create"] = mock_create
    try:
        with pytest.raises(RuntimeError, match="boom"):
            litellm_anthropic.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=16,
            )
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].status.description is not None
    assert "boom" in spans[0].status.description
    assert len(spans[0].events) == 1


def test_create_streaming(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]

    def mock_create(*args: Any, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        yield from _streaming_events("Streamed")

    LiteLLMInstrumentor.original_anthropic_funcs["create"] = mock_create
    try:
        stream = cast(
            Iterator[Dict[str, Any]],
            litellm_anthropic.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=16,
                stream=True,
            ),
        )
        chunks = list(stream)
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    assert len(chunks) == 6
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "messages.create"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Streamed"
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 8
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 4
    assert span.status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_acreate_streaming(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    original = LiteLLMInstrumentor.original_anthropic_funcs["acreate"]

    async def mock_acreate(*args: Any, **kwargs: Any) -> Any:
        async def _gen() -> AsyncIterator[Dict[str, Any]]:
            for event in _streaming_events("Async stream"):
                yield event

        return _gen()

    LiteLLMInstrumentor.original_anthropic_funcs["acreate"] = mock_acreate
    try:
        stream = cast(
            AsyncIterator[Dict[str, Any]],
            await litellm_anthropic.acreate(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=16,
                stream=True,
            ),
        )
        chunks = [chunk async for chunk in stream]
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["acreate"] = original

    assert len(chunks) == 6
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "Async stream"
    assert spans[0].status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_acreate_streaming_split_sse_bytes(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    payload = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in _streaming_events("Split bytes")
    ).encode()
    split_points = (17, 53, 121, 198, 257)
    chunks = [payload[start:end] for start, end in zip((0, *split_points), (*split_points, None))]
    original = LiteLLMInstrumentor.original_anthropic_funcs["acreate"]

    async def mock_acreate(*args: Any, **kwargs: Any) -> Any:
        async def _gen() -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        return _gen()

    LiteLLMInstrumentor.original_anthropic_funcs["acreate"] = mock_acreate
    try:
        stream = cast(
            AsyncIterator[bytes],
            await litellm_anthropic.acreate(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=16,
                stream=True,
            ),
        )
        assert [chunk async for chunk in stream] == chunks
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["acreate"] = original

    attributes = dict(
        cast(
            Mapping[str, AttributeValue], in_memory_span_exporter.get_finished_spans()[0].attributes
        )
    )
    assert attributes[SpanAttributes.OUTPUT_VALUE] == "Split bytes"
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 12


@pytest.mark.asyncio
async def test_create_returning_async_iterator(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]

    def mock_create(*args: Any, **kwargs: Any) -> AsyncIterator[Dict[str, Any]]:
        async def _gen() -> AsyncIterator[Dict[str, Any]]:
            for event in _streaming_events("Async from create"):
                yield event

        return _gen()

    LiteLLMInstrumentor.original_anthropic_funcs["create"] = mock_create
    try:
        stream = cast(
            AsyncIterator[Dict[str, Any]],
            litellm_anthropic.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=16,
                stream=True,
            ),
        )
        assert len([chunk async for chunk in stream]) == 6
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    attributes = dict(
        cast(
            Mapping[str, AttributeValue], in_memory_span_exporter.get_finished_spans()[0].attributes
        )
    )
    assert attributes[SpanAttributes.OUTPUT_VALUE] == "Async from create"


@pytest.mark.asyncio
async def test_create_returning_awaitable(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]

    async def mock_create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        return _mock_anthropic_response("Awaited")

    LiteLLMInstrumentor.original_anthropic_funcs["create"] = mock_create
    try:
        result = cast(
            Awaitable[Dict[str, Any]],
            litellm_anthropic.create(
                model=MODEL,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=16,
            ),
        )
        assert not in_memory_span_exporter.get_finished_spans()
        assert (await result)["content"][0]["text"] == "Awaited"
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    attributes = dict(
        cast(
            Mapping[str, AttributeValue], in_memory_span_exporter.get_finished_spans()[0].attributes
        )
    )
    assert attributes[SpanAttributes.OUTPUT_VALUE] == "Awaited"


def test_create_anthropic_cache_token_counts(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    response = _mock_anthropic_response()
    response["usage"] = {
        "input_tokens": 10,
        "output_tokens": 20,
        "cache_creation_input_tokens": 5,
        "cache_read_input_tokens": 3,
    }
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]
    LiteLLMInstrumentor.original_anthropic_funcs["create"] = lambda *args, **kwargs: response
    try:
        litellm_anthropic.create(
            model=MODEL,
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=16,
        )
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    attributes = dict(
        cast(
            Mapping[str, AttributeValue], in_memory_span_exporter.get_finished_spans()[0].attributes
        )
    )
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 18
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
    assert attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 38


def test_create_masking(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: TracerProvider,
) -> None:
    in_memory_span_exporter.clear()
    LiteLLMInstrumentor().instrument(
        tracer_provider=tracer_provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
    )
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]
    LiteLLMInstrumentor.original_anthropic_funcs["create"] = (
        lambda *args, **kwargs: _mock_anthropic_response("secret answer")
    )
    try:
        litellm_anthropic.create(
            model=MODEL,
            messages=[{"role": "user", "content": "secret question"}],
            max_tokens=16,
            system="secret system",
        )
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    attributes = dict(
        cast(
            Mapping[str, AttributeValue], in_memory_span_exporter.get_finished_spans()[0].attributes
        )
    )
    assert attributes[SpanAttributes.INPUT_VALUE] == REDACTED_VALUE
    assert attributes[SpanAttributes.OUTPUT_VALUE] == REDACTED_VALUE
    # Per-message content is dropped entirely when inputs/outputs are hidden.
    assert (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_CONTENT}"
        not in attributes
    )
    assert not any(key.startswith(SpanAttributes.LLM_OUTPUT_MESSAGES) for key in attributes)


def test_create_input_content_blocks(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
                },
            ],
        },
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]},
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_9", "content": "42"}],
        },
    ]
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]
    LiteLLMInstrumentor.original_anthropic_funcs["create"] = (
        lambda *args, **kwargs: _mock_anthropic_response()
    )
    try:
        litellm_anthropic.create(model=MODEL, messages=messages, max_tokens=16)
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    attributes = dict(
        cast(
            Mapping[str, AttributeValue], in_memory_span_exporter.get_finished_spans()[0].attributes
        )
    )
    in_prefix = SpanAttributes.LLM_INPUT_MESSAGES
    # image block
    assert (
        attributes[
            f"{in_prefix}.0.{MessageAttributes.MESSAGE_CONTENTS}.1."
            f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"
        ]
        == "image"
    )
    assert (
        attributes[
            f"{in_prefix}.0.{MessageAttributes.MESSAGE_CONTENTS}.1."
            f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
        ]
        == "data:image/png;base64,abc123"
    )
    # thinking block -> reasoning
    assert (
        attributes[
            f"{in_prefix}.1.{MessageAttributes.MESSAGE_CONTENTS}.0."
            f"{MessageContentAttributes.MESSAGE_CONTENT_TYPE}"
        ]
        == "reasoning"
    )
    assert (
        attributes[
            f"{in_prefix}.1.{MessageAttributes.MESSAGE_CONTENTS}.0."
            f"{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"
        ]
        == "hmm"
    )
    # tool_result block
    assert attributes[f"{in_prefix}.2.{MessageAttributes.MESSAGE_TOOL_CALL_ID}"] == "toolu_9"
    assert attributes[f"{in_prefix}.2.{MessageAttributes.MESSAGE_CONTENT}"] == "42"


def test_create_system_as_content_blocks(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()
    original = LiteLLMInstrumentor.original_anthropic_funcs["create"]
    LiteLLMInstrumentor.original_anthropic_funcs["create"] = (
        lambda *args, **kwargs: _mock_anthropic_response()
    )
    try:
        litellm_anthropic.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            system=[{"type": "text", "text": "sys instructions"}],  # type: ignore[arg-type]
        )
    finally:
        LiteLLMInstrumentor.original_anthropic_funcs["create"] = original

    attributes = dict(
        cast(
            Mapping[str, AttributeValue], in_memory_span_exporter.get_finished_spans()[0].attributes
        )
    )
    in_prefix = SpanAttributes.LLM_INPUT_MESSAGES
    assert attributes[f"{in_prefix}.0.{MessageAttributes.MESSAGE_ROLE}"] == "system"
    assert (
        attributes[
            f"{in_prefix}.0.{MessageAttributes.MESSAGE_CONTENTS}.0."
            f"{MessageContentAttributes.MESSAGE_CONTENT_TEXT}"
        ]
        == "sys instructions"
    )
    # Real user message is shifted to index 1.
    assert attributes[f"{in_prefix}.1.{MessageAttributes.MESSAGE_ROLE}"] == "user"


def test_uninstrument_anthropic(tracer_provider: TracerProvider) -> None:
    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider)
    assert getattr(litellm_anthropic.create, "is_wrapper", False)
    assert getattr(litellm_anthropic.messages.create, "is_wrapper", False)
    assert getattr(litellm_anthropic.acreate, "is_wrapper", False)

    instrumentor.uninstrument()
    assert litellm_anthropic.create.__name__ == "create"
    assert litellm_anthropic.messages.create.__name__ == "create"
    assert litellm_anthropic.acreate.__name__ == "acreate"
    assert not getattr(litellm_anthropic.create, "is_wrapper", False)

    instrumentor.instrument(tracer_provider=tracer_provider)
    assert getattr(litellm_anthropic.create, "is_wrapper", False)


def test_oitracer(setup_litellm_instrumentation: Any) -> None:
    assert isinstance(LiteLLMInstrumentor()._tracer, OITracer)


@pytest.fixture()
def session_id() -> str:
    return "my-test-session-id"


@pytest.fixture()
def user_id() -> str:
    return "my-test-user-id"


@pytest.fixture()
def metadata() -> Dict[str, Any]:
    return {
        "test-int": 1,
        "test-str": "string",
        "test-list": [1, 2, 3],
        "test-dict": {"key": "value"},
    }


@pytest.fixture()
def tags() -> List[str]:
    return ["tag-1", "tag-2"]


@pytest.fixture()
def prompt_template() -> str:
    return "This is a test prompt template with int: {test_int}"


@pytest.fixture()
def prompt_template_version() -> str:
    return "v1.0"


@pytest.fixture()
def prompt_template_variables() -> Dict[str, Any]:
    return {"test_int": 1, "test_str": "string", "test_bool": True}
