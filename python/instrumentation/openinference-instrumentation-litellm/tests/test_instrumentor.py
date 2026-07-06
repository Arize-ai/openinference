import json
import os
from typing import Any, Dict, Generator, List, Mapping, Optional, Union, cast
from unittest.mock import patch

import litellm
import pytest
from litellm import OpenAIChatCompletion  # type: ignore[attr-defined, unused-ignore]
from litellm.types.utils import EmbeddingResponse, ImageObject, ImageResponse, Usage
from litellm.types.utils import Message as LitellmMessage
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from opentelemetry.util._importlib_metadata import entry_points
from opentelemetry.util.types import AttributeValue

from openinference.instrumentation import OITracer, safe_json_dumps, using_attributes
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from openinference.semconv.trace import (
    EmbeddingAttributes,
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)

OUTPUT_VALUE = SpanAttributes.OUTPUT_VALUE


@pytest.fixture
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.fixture()
def setup_litellm_instrumentation(
    tracer_provider: TracerProvider,
) -> Generator[None, None, None]:
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    yield


class TestInstrumentor:
    def test_entrypoint_for_opentelemetry_instrument(self) -> None:
        (instrumentor_entrypoint,) = entry_points(
            group="opentelemetry_instrumentor", name="litellm"
        )
        instrumentor = instrumentor_entrypoint.load()()
        assert isinstance(instrumentor, LiteLLMInstrumentor)

    # Ensure we're using the common OITracer from common openinference-instrumentation pkg
    def test_oitracer(self, setup_litellm_instrumentation: Any) -> None:
        assert isinstance(LiteLLMInstrumentor()._tracer, OITracer)


@pytest.mark.parametrize("use_context_attributes", [False, True])
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize(
    "input_messages",
    [
        [{"content": "What's the capital of China?", "role": "user"}],
        [LitellmMessage(content="How can I help you?", role="assistant")],
    ],
)
def test_completion(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    input_messages: List[Union[Dict[str, Any], LitellmMessage]],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
    n: int,
) -> None:
    in_memory_span_exporter.clear()

    response = None
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=input_messages,
                n=n,
                mock_response="Beijing",
            )
    else:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            n=n,
            mock_response="Beijing",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    input_values = [
        msg.json() if isinstance(msg, LitellmMessage) else msg  # type: ignore[no-untyped-call]
        for msg in input_messages
    ]
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps({"messages": input_values})
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert str(attributes.get(SpanAttributes.OUTPUT_VALUE)) == "Beijing"
    for i, choice in enumerate(response["choices"]):
        _check_llm_message(SpanAttributes.LLM_OUTPUT_MESSAGES, i, attributes, choice.message)

    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


@pytest.mark.parametrize("use_context_attributes", [True])
@pytest.mark.parametrize("n", [1])
def test_completion_sync_streaming(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
    n: int,
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
    response = None
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=input_messages,
                mock_response="The capital of China is Beijing",
                n=n,
                stream=True,
            )
    else:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            mock_response="The capital of China is Beijing",
            n=n,
            stream=True,
        )

    output_message = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            output_message += chunk.choices[0].delta.content

    assert output_message == "The capital of China is Beijing"

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "The capital of China is Beijing"

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


def test_completion_with_parameters(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=input_messages,
        mock_response="Beijing",
        temperature=0.7,
        top_p=0.9,
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS) == json.dumps(
        {
            "model": "gpt-3.5-turbo",
            "mock_response": "Beijing",
            "temperature": 0.7,
            "top_p": 0.9,
        }
    )

    assert "Beijing" == attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert span.status.status_code == StatusCode.OK


def test_completion_with_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    input_messages: List[Dict[str, Any]] = [
        {"content": "What's the weather like in New York?", "role": "user"},
        {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "New York", "unit": "celsius"}',
                    },
                }
            ],
        },
    ]
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=input_messages,
        mock_response="The weather in New York is 22°C and sunny.",
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

    for i, message in enumerate(input_messages):
        _check_llm_message(SpanAttributes.LLM_INPUT_MESSAGES, i, attributes, message)

    tool_call_function_name = (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    )
    tool_call_function_args = (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )

    assert attributes.get(tool_call_function_name) == "get_weather"
    assert attributes.get(tool_call_function_args) == '{"location": "New York", "unit": "celsius"}'

    assert "The weather in New York is 22°C and sunny." == attributes.get(OUTPUT_VALUE)


def test_completion_streaming_with_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Test that tool calls are captured correctly in streaming responses."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    in_memory_span_exporter.clear()

    # Helper to create Delta with tool_calls (needs model_construct for extra fields)
    def make_delta(
        role: Optional[str] = None,
        content: Optional[str] = None,
        tool_calls: Optional[List[ChoiceDeltaToolCall]] = None,
    ) -> Delta:
        return Delta.model_construct(role=role, content=content, tool_calls=tool_calls)

    chunks = [
        ModelResponseStream(
            id="chatcmpl-123",
            model="gpt-3.5-turbo",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=make_delta(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_abc123",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="get_weather",
                                    arguments="",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-123",
            model="gpt-3.5-turbo",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=make_delta(
                        role=None,
                        content=None,
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=None,
                                type=None,
                                function=ChoiceDeltaToolCallFunction(
                                    name=None,
                                    arguments='{"location": "New ',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-123",
            model="gpt-3.5-turbo",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=make_delta(
                        role=None,
                        content=None,
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=None,
                                type=None,
                                function=ChoiceDeltaToolCallFunction(
                                    name=None,
                                    arguments='York"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-123",
            model="gpt-3.5-turbo",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=make_delta(role=None, content=None, tool_calls=None),
                    finish_reason="tool_calls",
                )
            ],
        ),
    ]

    class MockStreamWrapper:
        def __init__(self, chunks: List[ModelResponseStream]) -> None:
            self._chunks = chunks
            self._index = 0

        def __iter__(self) -> "MockStreamWrapper":
            return self

        def __next__(self) -> ModelResponseStream:
            if self._index >= len(self._chunks):
                raise StopIteration
            chunk: ModelResponseStream = self._chunks[self._index]
            self._index += 1
            return chunk

    input_messages = [{"content": "What's the weather like in New York?", "role": "user"}]

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]

    # The wrapper does `from litellm.litellm_core_utils.streaming_handler import
    # CustomStreamWrapper` on each call, so patching the source module makes the
    # `isinstance(result, CustomStreamWrapper)` branch accept our mock.
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = (
            lambda *args, **kwargs: MockStreamWrapper(chunks)
        )

        with patch(
            "litellm.litellm_core_utils.streaming_handler.CustomStreamWrapper",
            MockStreamWrapper,
        ):
            response = litellm.completion(
                model="gpt-3.5-turbo",
                messages=input_messages,
                stream=True,
            )

            chunks_received = list(response)
            assert len(chunks_received) == 4
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    tool_call_function_name = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    )
    tool_call_function_args = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )

    assert attributes.get(tool_call_function_name) == "get_weather"
    assert attributes.get(tool_call_function_args) == '{"location": "New York"}'
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) is None


@pytest.mark.asyncio
async def test_acompletion_streaming_with_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Test that tool calls are captured correctly in async streaming responses."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
    from openai.types.chat.chat_completion_chunk import (
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    in_memory_span_exporter.clear()

    def make_delta(
        role: Optional[str] = None,
        content: Optional[str] = None,
        tool_calls: Optional[List[ChoiceDeltaToolCall]] = None,
    ) -> Delta:
        return Delta.model_construct(role=role, content=content, tool_calls=tool_calls)

    chunks = [
        ModelResponseStream(
            id="chatcmpl-456",
            model="gpt-3.5-turbo",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=make_delta(
                        role="assistant",
                        content=None,
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_def456",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="search",
                                    arguments="",
                                ),
                            ),
                            ChoiceDeltaToolCall(
                                index=1,
                                id="call_ghi789",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="calculate",
                                    arguments="",
                                ),
                            ),
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-456",
            model="gpt-3.5-turbo",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=make_delta(
                        role=None,
                        content=None,
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id=None,
                                type=None,
                                function=ChoiceDeltaToolCallFunction(
                                    name=None,
                                    arguments='{"query": "test"}',
                                ),
                            ),
                            ChoiceDeltaToolCall(
                                index=1,
                                id=None,
                                type=None,
                                function=ChoiceDeltaToolCallFunction(
                                    name=None,
                                    arguments='{"x": 1, "y": 2}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-456",
            model="gpt-3.5-turbo",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=make_delta(role=None, content=None, tool_calls=None),
                    finish_reason="tool_calls",
                )
            ],
        ),
    ]

    class MockAsyncStreamWrapper:
        def __init__(self, chunks: List[ModelResponseStream]) -> None:
            self._chunks = chunks
            self._index = 0

        def __aiter__(self) -> "MockAsyncStreamWrapper":
            return self

        async def __anext__(self) -> ModelResponseStream:
            if self._index >= len(self._chunks):
                raise StopAsyncIteration
            chunk: ModelResponseStream = self._chunks[self._index]
            self._index += 1
            return chunk

    input_messages = [{"content": "Search for test and calculate 1+2", "role": "user"}]

    original_func = LiteLLMInstrumentor.original_litellm_funcs["acompletion"]

    async def mock_acompletion(*args: Any, **kwargs: Any) -> MockAsyncStreamWrapper:
        return MockAsyncStreamWrapper(chunks)

    try:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = mock_acompletion

        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            stream=True,
        )

        chunks_received = [chunk async for chunk in response]
        assert len(chunks_received) == 3
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    tool_call_0_name = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    )
    tool_call_0_args = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    assert attributes.get(tool_call_0_name) == "search"
    assert attributes.get(tool_call_0_args) == '{"query": "test"}'

    tool_call_1_name = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.1."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    )
    tool_call_1_args = (
        f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.1."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
    )
    assert attributes.get(tool_call_1_name) == "calculate"
    assert attributes.get(tool_call_1_args) == '{"x": 1, "y": 2}'


def _assert_no_message_content_id(attributes: Mapping[str, AttributeValue]) -> None:
    assert not any(
        key.endswith(f".{MessageContentAttributes.MESSAGE_CONTENT_ID}") for key in attributes
    )


def _reasoning_content_key(prefix: str, message_index: int, content_index: int, attr: str) -> str:
    return f"{prefix}.{message_index}.{MessageAttributes.MESSAGE_CONTENTS}.{content_index}.{attr}"


def _pop_reasoning_content(
    attributes: Dict[str, AttributeValue],
    prefix: str,
    message_index: int,
    content_index: int,
    *,
    text: Optional[str] = None,
    signature: Optional[str] = None,
    data: Optional[str] = None,
    encrypted_content: Optional[str] = None,
) -> None:
    """Pop and validate a single reasoning ``message_content`` block. Only the
    fields passed in are expected; the block type is always asserted."""

    def key(attr: str) -> str:
        return _reasoning_content_key(prefix, message_index, content_index, attr)

    assert attributes.pop(key(MessageContentAttributes.MESSAGE_CONTENT_TYPE)) == "reasoning"
    if text is not None:
        assert attributes.pop(key(MessageContentAttributes.MESSAGE_CONTENT_TEXT)) == text
    if signature is not None:
        assert attributes.pop(key(MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE)) == signature
    if data is not None:
        assert attributes.pop(key(MessageContentAttributes.MESSAGE_CONTENT_DATA)) == data
    if encrypted_content is not None:
        assert (
            attributes.pop(key(MessageContentAttributes.MESSAGE_CONTENT_ENCRYPTED_CONTENT))
            == encrypted_content
        )


def _pop_text_content(
    attributes: Dict[str, AttributeValue],
    prefix: str,
    message_index: int,
    content_index: int,
    text: str,
) -> None:
    """Pop and validate a single text ``message_content`` block."""
    key = _reasoning_content_key(
        prefix, message_index, content_index, MessageContentAttributes.MESSAGE_CONTENT_TYPE
    )
    assert attributes.pop(key) == "text"
    key = _reasoning_content_key(
        prefix, message_index, content_index, MessageContentAttributes.MESSAGE_CONTENT_TEXT
    )
    assert attributes.pop(key) == text


def _pop_tool_use_content(
    attributes: Dict[str, AttributeValue],
    prefix: str,
    message_index: int,
    content_index: int,
    *,
    tool_call_id: str,
    name: str,
    arguments: str,
) -> None:
    """Pop and validate a single tool_use ``message_content`` block."""

    def key(attr: str) -> str:
        return _reasoning_content_key(prefix, message_index, content_index, attr)

    assert attributes.pop(key(MessageContentAttributes.MESSAGE_CONTENT_TYPE)) == "tool_use"
    assert attributes.pop(key(ToolCallAttributes.TOOL_CALL_ID)) == tool_call_id
    assert attributes.pop(key(ToolCallAttributes.TOOL_CALL_FUNCTION_NAME)) == name
    assert attributes.pop(key(ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON)) == arguments


def _make_thinking_blocks_response() -> Any:
    from litellm.types.utils import Choices, ModelResponse, Usage

    return ModelResponse(
        id="chatcmpl-thinking",
        model="claude-sonnet-4-5",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=LitellmMessage(
                    role="assistant",
                    content="Paris.",
                    reasoning_content="First thought.Second thought.",
                    thinking_blocks=[
                        {
                            "type": "thinking",
                            "thinking": "First thought.",
                            "signature": "sig-first",
                        },
                        {
                            "type": "thinking",
                            "thinking": "Second thought.",
                            "signature": "sig-second",
                        },
                    ],
                ),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _assert_thinking_blocks_attributes(
    attributes: Dict[str, AttributeValue],
    input_messages: List[Dict[str, Any]],
) -> None:
    """Exhaustively pop and validate every attribute produced for the
    thinking-blocks mock response; nothing unexpected may remain."""
    _assert_no_message_content_id(attributes)
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "anthropic/claude-sonnet-4-5"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "anthropic"
    assert attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS) == safe_json_dumps(
        {"model": "anthropic/claude-sonnet-4-5"}
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Paris."

    in_prefix = SpanAttributes.LLM_INPUT_MESSAGES
    assert attributes.pop(f"{in_prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{in_prefix}.0.{MessageAttributes.MESSAGE_CONTENT}")
        == "What's the capital of France?"
    )

    out_prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
    assert attributes.pop(f"{out_prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    _pop_reasoning_content(
        attributes, out_prefix, 0, 0, text="First thought.", signature="sig-first"
    )
    _pop_reasoning_content(
        attributes, out_prefix, 0, 1, text="Second thought.", signature="sig-second"
    )
    _pop_text_content(attributes, out_prefix, 0, 2, "Paris.")

    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert attributes == {}


def test_completion_with_thinking_blocks(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Non-streaming completion captures thinking_blocks as ordered reasoning contents."""
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of France?", "role": "user"}]
    mock_response = _make_thinking_blocks_response()

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = (
            lambda *args, **kwargs: mock_response
        )
        litellm.completion(
            model="anthropic/claude-sonnet-4-5",
            messages=input_messages,
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    _assert_thinking_blocks_attributes(attributes, input_messages)


@pytest.mark.asyncio
async def test_acompletion_with_thinking_blocks(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Non-streaming acompletion captures thinking_blocks as ordered reasoning contents."""
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of France?", "role": "user"}]
    mock_response = _make_thinking_blocks_response()

    async def mock_acompletion(*args: Any, **kwargs: Any) -> Any:
        return mock_response

    original_func = LiteLLMInstrumentor.original_litellm_funcs["acompletion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = mock_acompletion
        await litellm.acompletion(
            model="anthropic/claude-sonnet-4-5",
            messages=input_messages,
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    _assert_thinking_blocks_attributes(attributes, input_messages)


def test_completion_with_reasoning_content_only(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """reasoning_content without thinking_blocks yields a single reasoning content."""
    from litellm.types.utils import Choices, ModelResponse, Usage

    in_memory_span_exporter.clear()

    mock_response = ModelResponse(
        id="chatcmpl-reasoning",
        model="deepseek-reasoner",
        choices=[
            Choices(
                index=0,
                finish_reason="stop",
                message=LitellmMessage(
                    role="assistant",
                    content="Paris.",
                    reasoning_content="The user asks about France.",
                ),
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = (
            lambda *args, **kwargs: mock_response
        )
        litellm.completion(
            model="deepseek/deepseek-reasoner",
            messages=[{"content": "What's the capital of France?", "role": "user"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    _assert_no_message_content_id(attributes)

    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "deepseek/deepseek-reasoner"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "deepseek"
    assert attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS) == safe_json_dumps(
        {"model": "deepseek/deepseek-reasoner"}
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": [{"content": "What's the capital of France?", "role": "user"}]}
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Paris."

    in_prefix = SpanAttributes.LLM_INPUT_MESSAGES
    assert attributes.pop(f"{in_prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{in_prefix}.0.{MessageAttributes.MESSAGE_CONTENT}")
        == "What's the capital of France?"
    )

    out_prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
    assert attributes.pop(f"{out_prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    # A single reasoning block with no signature, followed by the text content
    _pop_reasoning_content(attributes, out_prefix, 0, 0, text="The user asks about France.")
    _pop_text_content(attributes, out_prefix, 0, 1, "Paris.")

    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert attributes == {}


def test_completion_tool_flow_with_input_thinking_blocks(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Assistant input messages carrying thinking_blocks in a tool-calling flow
    are captured as reasoning contents alongside existing tool call attributes."""
    in_memory_span_exporter.clear()

    input_messages: List[Dict[str, Any]] = [
        {"content": "What's the weather like in New York?", "role": "user"},
        {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "reasoning_content": "I need to call the weather tool.",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I need to call the weather tool.",
                    "signature": "sig-tool-flow",
                }
            ],
            "tool_calls": [
                {
                    "index": 1,
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "New York"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": "22C and sunny",
        },
    ]
    litellm.completion(
        model="anthropic/claude-sonnet-4-5",
        messages=input_messages,
        mock_response="The weather in New York is 22C and sunny.",
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    _assert_no_message_content_id(attributes)

    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "anthropic/claude-sonnet-4-5"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "anthropic"
    assert attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS) == safe_json_dumps(
        {
            "model": "anthropic/claude-sonnet-4-5",
            "mock_response": "The weather in New York is 22C and sunny.",
        }
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert (
        attributes.pop(SpanAttributes.OUTPUT_VALUE) == "The weather in New York is 22C and sunny."
    )

    prefix = SpanAttributes.LLM_INPUT_MESSAGES
    assert attributes.pop(f"{prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{prefix}.0.{MessageAttributes.MESSAGE_CONTENT}")
        == "What's the weather like in New York?"
    )

    # Assistant message with reasoning: ordered contents (reasoning -> text -> tool_use)
    assert attributes.pop(f"{prefix}.1.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    _pop_reasoning_content(
        attributes, prefix, 1, 0, text="I need to call the weather tool.", signature="sig-tool-flow"
    )
    _pop_text_content(attributes, prefix, 1, 1, "Let me check the weather for you.")
    _pop_tool_use_content(
        attributes,
        prefix,
        1,
        2,
        tool_call_id="call_abc123",
        name="get_weather",
        arguments='{"location": "New York"}',
    )
    tool_call_prefix = f"{prefix}.1.{MessageAttributes.MESSAGE_TOOL_CALLS}.0"
    assert attributes.pop(f"{tool_call_prefix}.{ToolCallAttributes.TOOL_CALL_ID}") == "call_abc123"
    assert (
        attributes.pop(f"{tool_call_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}")
        == "get_weather"
    )
    assert (
        attributes.pop(f"{tool_call_prefix}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}")
        == '{"location": "New York"}'
    )

    assert attributes.pop(f"{prefix}.2.{MessageAttributes.MESSAGE_ROLE}") == "tool"
    assert attributes.pop(f"{prefix}.2.{MessageAttributes.MESSAGE_CONTENT}") == "22C and sunny"

    out_prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
    assert attributes.pop(f"{out_prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    assert (
        attributes.pop(f"{out_prefix}.0.{MessageAttributes.MESSAGE_CONTENT}")
        == "The weather in New York is 22C and sunny."
    )

    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.pop(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert cast(float, attributes.pop(SpanAttributes.LLM_COST_TOTAL)) > 0
    assert attributes == {}


def _make_reasoning_stream_chunks() -> List[Any]:
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    return [
        ModelResponseStream(
            id="chatcmpl-reasoning-stream",
            model="claude-sonnet-4-5",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        role="assistant",
                        content=None,
                        reasoning_content="First ",
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "First ", "signature": ""}
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-reasoning-stream",
            model="claude-sonnet-4-5",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        reasoning_content="thought.",
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "thought.", "signature": ""}
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-reasoning-stream",
            model="claude-sonnet-4-5",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "", "signature": "sig-stream"}
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-reasoning-stream",
            model="claude-sonnet-4-5",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=None,
                        thinking_blocks=[{"type": "redacted_thinking", "data": "redacted-data"}],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-reasoning-stream",
            model="claude-sonnet-4-5",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content="Paris."),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-reasoning-stream",
            model="claude-sonnet-4-5",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(content=None),
                    finish_reason="stop",
                )
            ],
        ),
    ]


def _assert_streaming_reasoning_attributes(attributes: Dict[str, AttributeValue]) -> None:
    """Exhaustively pop and validate every attribute produced for the reasoning
    stream chunks; nothing unexpected may remain."""
    _assert_no_message_content_id(attributes)
    assert attributes.pop(SpanAttributes.OPENINFERENCE_SPAN_KIND) == "LLM"
    assert attributes.pop(SpanAttributes.LLM_MODEL_NAME) == "anthropic/claude-sonnet-4-5"
    assert attributes.pop(SpanAttributes.LLM_PROVIDER) == "anthropic"
    assert attributes.pop(SpanAttributes.LLM_INVOCATION_PARAMETERS) == safe_json_dumps(
        {"model": "anthropic/claude-sonnet-4-5", "stream": True}
    )
    assert attributes.pop(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": [{"content": "What's the capital of France?", "role": "user"}]}
    )
    assert attributes.pop(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    assert attributes.pop(SpanAttributes.OUTPUT_VALUE) == "Paris."

    in_prefix = SpanAttributes.LLM_INPUT_MESSAGES
    assert attributes.pop(f"{in_prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "user"
    assert (
        attributes.pop(f"{in_prefix}.0.{MessageAttributes.MESSAGE_CONTENT}")
        == "What's the capital of France?"
    )

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
    assert attributes.pop(f"{prefix}.0.{MessageAttributes.MESSAGE_ROLE}") == "assistant"
    # Ordered contents: thinking block, redacted block, then text
    _pop_reasoning_content(attributes, prefix, 0, 0, text="First thought.", signature="sig-stream")
    _pop_reasoning_content(attributes, prefix, 0, 1, data="redacted-data")
    _pop_text_content(attributes, prefix, 0, 2, "Paris.")
    assert attributes == {}


def test_completion_streaming_with_reasoning(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Sync streaming accumulates delta.reasoning_content and delta.thinking_blocks."""
    from litellm.types.utils import ModelResponseStream

    in_memory_span_exporter.clear()

    chunks = _make_reasoning_stream_chunks()

    class MockStreamWrapper:
        def __init__(self, chunks: List[ModelResponseStream]) -> None:
            self._chunks = chunks
            self._index = 0

        def __iter__(self) -> "MockStreamWrapper":
            return self

        def __next__(self) -> ModelResponseStream:
            if self._index >= len(self._chunks):
                raise StopIteration
            chunk: ModelResponseStream = self._chunks[self._index]
            self._index += 1
            return chunk

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = (
            lambda *args, **kwargs: MockStreamWrapper(chunks)
        )
        with patch(
            "litellm.litellm_core_utils.streaming_handler.CustomStreamWrapper",
            MockStreamWrapper,
        ):
            response = litellm.completion(
                model="anthropic/claude-sonnet-4-5",
                messages=[{"content": "What's the capital of France?", "role": "user"}],
                stream=True,
            )
            chunks_received = list(response)
            assert len(chunks_received) == 6
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    _assert_streaming_reasoning_attributes(attributes)


@pytest.mark.asyncio
async def test_acompletion_streaming_with_reasoning(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Async streaming accumulates delta.reasoning_content and delta.thinking_blocks."""
    from litellm.types.utils import ModelResponseStream

    in_memory_span_exporter.clear()

    chunks = _make_reasoning_stream_chunks()

    class MockAsyncStreamWrapper:
        def __init__(self, chunks: List[ModelResponseStream]) -> None:
            self._chunks = chunks
            self._index = 0

        def __aiter__(self) -> "MockAsyncStreamWrapper":
            return self

        async def __anext__(self) -> ModelResponseStream:
            if self._index >= len(self._chunks):
                raise StopAsyncIteration
            chunk: ModelResponseStream = self._chunks[self._index]
            self._index += 1
            return chunk

    original_func = LiteLLMInstrumentor.original_litellm_funcs["acompletion"]

    async def mock_acompletion(*args: Any, **kwargs: Any) -> MockAsyncStreamWrapper:
        return MockAsyncStreamWrapper(chunks)

    try:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = mock_acompletion
        response = await litellm.acompletion(
            model="anthropic/claude-sonnet-4-5",
            messages=[{"content": "What's the capital of France?", "role": "user"}],
            stream=True,
        )
        chunks_received = [chunk async for chunk in response]
        assert len(chunks_received) == 6
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], spans[0].attributes))
    _assert_streaming_reasoning_attributes(attributes)


def test_completion_with_tool_schema_capture(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_forecast",
                "description": "Get weather forecast for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "days": {"type": "integer", "minimum": 1, "maximum": 7},
                    },
                    "required": ["location", "days"],
                },
            },
        },
    ]

    input_messages = [{"content": "What's the weather like in New York?", "role": "user"}]
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=input_messages,
        tools=tools,
        tool_choice="auto",
        mock_response="I'll check the weather for you.",
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    # Verify basic attributes
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

    # Verify tool schemas are captured
    tool1_schema = attributes.get(f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}")
    tool2_schema = attributes.get(f"{SpanAttributes.LLM_TOOLS}.1.{ToolAttributes.TOOL_JSON_SCHEMA}")

    assert tool1_schema is not None
    assert tool2_schema is not None
    assert isinstance(tool1_schema, str)
    assert isinstance(tool2_schema, str)

    # Verify first tool schema
    tool1_schema_dict = json.loads(tool1_schema)
    assert tool1_schema_dict["type"] == "function"
    assert tool1_schema_dict["function"]["name"] == "get_weather"
    assert tool1_schema_dict["function"]["description"] == "Get current weather in a given location"
    assert tool1_schema_dict["function"]["parameters"]["properties"]["location"]["type"] == "string"
    assert tool1_schema_dict["function"]["parameters"]["properties"]["unit"]["enum"] == [
        "celsius",
        "fahrenheit",
    ]
    assert tool1_schema_dict["function"]["parameters"]["required"] == ["location"]

    # Verify second tool schema
    tool2_schema_dict = json.loads(tool2_schema)
    assert tool2_schema_dict["type"] == "function"
    assert tool2_schema_dict["function"]["name"] == "get_forecast"
    assert tool2_schema_dict["function"]["description"] == "Get weather forecast for a location"
    assert tool2_schema_dict["function"]["parameters"]["properties"]["location"]["type"] == "string"
    assert tool2_schema_dict["function"]["parameters"]["properties"]["days"]["type"] == "integer"
    assert tool2_schema_dict["function"]["parameters"]["required"] == ["location", "days"]

    assert "I'll check the weather for you." == attributes.get(OUTPUT_VALUE)


async def test_acompletion_with_tool_schema_capture(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Test that async completion captures tool schemas correctly"""
    in_memory_span_exporter.clear()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get current time in a timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "Timezone name"},
                    },
                    "required": ["timezone"],
                },
            },
        }
    ]

    input_messages = [{"content": "What time is it in Tokyo?", "role": "user"}]
    await litellm.acompletion(
        model="gpt-3.5-turbo",
        messages=input_messages,
        tools=tools,
        mock_response="I'll check the time in Tokyo for you.",
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    # Verify tool schema is captured
    tool_schema = attributes.get(f"{SpanAttributes.LLM_TOOLS}.0.{ToolAttributes.TOOL_JSON_SCHEMA}")
    assert tool_schema is not None

    tool_schema_dict = json.loads(cast(str, tool_schema))
    assert tool_schema_dict["function"]["name"] == "get_time"
    assert tool_schema_dict["function"]["description"] == "Get current time in a timezone"

    assert "I'll check the time in Tokyo for you." == attributes.get(OUTPUT_VALUE)


def test_completion_with_multiple_messages(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [
        {"content": "Hello, I want to bake a cake", "role": "user"},
        {"content": "Hello, I can pull up some recipes for cakes.", "role": "assistant"},
        {"content": "No actually I want to make a pie", "role": "user"},
    ]
    litellm.completion(
        model="gpt-3.5-turbo",
        messages=input_messages,
        mock_response="Got it! What kind of pie would you like to make?",
        api_key=os.getenv("OPENAI_API_KEY", "sk-"),
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    for i, message in enumerate(input_messages):
        _check_llm_message(SpanAttributes.LLM_INPUT_MESSAGES, i, attributes, message)
    assert attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS) == json.dumps(
        {
            "model": "gpt-3.5-turbo",
            "mock_response": "Got it! What kind of pie would you like to make?",
        }
    )
    assert "Got it! What kind of pie would you like to make?" == attributes.get(OUTPUT_VALUE)
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert span.status.status_code == StatusCode.OK


def test_completion_image_support(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://dummy_image.jpg"},
                },
            ],
        }
    ]
    litellm.completion(
        model="gpt-4o",
        messages=input_messages,
        mock_response="That's an image of a pasture",
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-4o"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"
    for i, message in enumerate(input_messages):
        _check_llm_message(SpanAttributes.LLM_INPUT_MESSAGES, i, attributes, message)
    params_str = attributes.get(SpanAttributes.LLM_INVOCATION_PARAMETERS)
    assert isinstance(params_str, str)  # Type narrowing for mypy
    assert json.loads(params_str) == {
        "model": "gpt-4o",
        "mock_response": "That's an image of a pasture",
    }
    assert "That's an image of a pasture" == attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert span.status.status_code == StatusCode.OK


def test_completion_with_invalid_model_triggers_exception_event(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with pytest.raises(Exception):
        litellm.completion(
            model="invalid-model-name",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, "Expected one span to be recorded"

    span = spans[0]
    assert span.name == "completion"
    assert span.status.status_code == StatusCode.ERROR

    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1, "Expected one exception event to be recorded"

    exception_attributes = cast(Mapping[str, AttributeValue], exception_events[0].attributes)
    assert "exception.type" in exception_attributes
    assert "exception.message" in exception_attributes
    assert "exception.stacktrace" in exception_attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_acompletion(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=input_messages,
                mock_response="Beijing",
            )
    else:
        await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            mock_response="Beijing",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

    assert "Beijing" == attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_acompletion_stream(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=input_messages,
                mock_response="Beijing",
                stream=True,
            )
            async for chunk in response:
                print(chunk)
    else:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            mock_response="Beijing",
            stream=True,
        )
        async for chunk in response:
            print(chunk)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

    assert "Beijing" == attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_acompletion_stream_token_count(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            response = await litellm.acompletion(
                model="gpt-3.5-turbo",
                messages=input_messages,
                mock_response="Beijing",
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in response:
                print(chunk)
    else:
        response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=input_messages,
            mock_response="Beijing",
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in response:
            print(chunk)

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "acompletion"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

    assert "Beijing" == attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 14
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 2
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 16
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


async def test_acompletion_with_invalid_model_triggers_exception_event(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with pytest.raises(Exception):
        await litellm.acompletion(
            model="invalid-model-name",
            messages=[{"content": "What's the capital of China?", "role": "user"}],
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, "Expected one span to be recorded"

    span = spans[0]
    assert span.name == "acompletion"
    assert span.status.status_code == StatusCode.ERROR

    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1, "Expected one exception event to be recorded"

    exception_attributes = cast(Mapping[str, AttributeValue], exception_events[0].attributes)
    assert "exception.type" in exception_attributes
    assert "exception.message" in exception_attributes
    assert "exception.stacktrace" in exception_attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_completion_with_retries(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    input_messages = [{"content": "What's the capital of China?", "role": "user"}]
    if use_context_attributes:
        with using_attributes(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
            tags=tags,
            prompt_template=prompt_template,
            prompt_template_version=prompt_template_version,
            prompt_template_variables=prompt_template_variables,
        ):
            litellm.completion_with_retries(  # type: ignore [no-untyped-call]
                model="gpt-3.5-turbo",
                messages=input_messages,
                mock_response="Beijing",
            )
    else:
        litellm.completion_with_retries(  # type: ignore [no-untyped-call]
            model="gpt-3.5-turbo",
            messages=input_messages,
            mock_response="Beijing",
        )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "completion_with_retries"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "gpt-3.5-turbo"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == safe_json_dumps(
        {"messages": input_messages}
    )
    assert attributes.get(SpanAttributes.INPUT_MIME_TYPE) == "application/json"

    assert "Beijing" == attributes.get(SpanAttributes.OUTPUT_VALUE)
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 10
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 20
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 30
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


# Bug report filed on GitHub for acompletion_with_retries: https://github.com/BerriAI/litellm/issues/4908
# Until litellm fixes acompletion_with_retries keep this test commented
# async def test_acompletion_with_retries(tracer_provider, in_memory_span_exporter):
#     in_memory_span_exporter.clear()
#
#     await litellm.acompletion_with_retries(
#         model="gpt-3.5-turbo",
#         messages=[{"content": "What's the capital of China?", "role": "user"}],
#     )
#     spans = in_memory_span_exporter.get_finished_spans()
#     assert len(spans) == 1
#     span = spans[0]
#     assert span.name == "acompletion_with_retries"
#     assert span.attributes[SpanAttributes.LLM_MODEL_NAME] == "gpt-3.5-turbo"
#     assert span.attributes[SpanAttributes.INPUT_VALUE] == "What's the capital of China?"

#     assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "Beijing"
#     assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] == 10
#     assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] == 20
#     assert span.attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] == 30

# Unlike the completion() functions, liteLLM does not offer a mock_response parameter
# for embeddings or image gen yet
# For now the following tests monkeypatch OpenAIChatCompletion functions


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_embedding(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    mock_response_embedding = EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        object="list",
        usage=Usage(prompt_tokens=6, completion_tokens=1, total_tokens=6),
    )

    with patch.object(OpenAIChatCompletion, "embedding", return_value=mock_response_embedding):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                litellm.embedding(
                    model="text-embedding-ada-002", input=["good morning from litellm"]
                )
        else:
            litellm.embedding(model="text-embedding-ada-002", input=["good morning from litellm"])

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "CreateEmbeddings"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-ada-002"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == str(["good morning from litellm"])
    assert (
        attributes.get(SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS)
        == '{"model": "text-embedding-ada-002"}'
    )

    assert (
        attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "good morning from litellm"
    )
    assert attributes.get(
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    ) == (0.1, 0.2, 0.3)
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 6
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 1
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 6
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


def test_embedding_with_invalid_model_triggers_exception_event(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with pytest.raises(Exception):
        litellm.embedding(model="invalid-model-name", input=["good morning from litellm"])

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, "Expected one span to be recorded"

    span = spans[0]
    assert span.name == "CreateEmbeddings"
    assert span.status.status_code == StatusCode.ERROR

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert (
        attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "good morning from litellm"
    )

    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1, "Expected one exception event to be recorded"

    exception_attributes = cast(Mapping[str, AttributeValue], exception_events[0].attributes)
    assert "exception.type" in exception_attributes
    assert "exception.message" in exception_attributes
    assert "exception.stacktrace" in exception_attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_aembedding(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    mock_response_embedding = EmbeddingResponse(
        model="text-embedding-ada-002",
        data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
        object="list",
        usage=Usage(prompt_tokens=6, completion_tokens=1, total_tokens=6),
    )

    with patch.object(OpenAIChatCompletion, "aembedding", return_value=mock_response_embedding):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                await litellm.aembedding(
                    model="text-embedding-ada-002", input=["good morning from litellm"]
                )
        else:
            await litellm.aembedding(
                model="text-embedding-ada-002", input=["good morning from litellm"]
            )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "CreateEmbeddings"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.EMBEDDING_MODEL_NAME) == "text-embedding-ada-002"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == str(["good morning from litellm"])
    assert (
        attributes.get(SpanAttributes.EMBEDDING_INVOCATION_PARAMETERS)
        == '{"model": "text-embedding-ada-002"}'
    )

    assert (
        attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "good morning from litellm"
    )
    assert attributes.get(
        f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_VECTOR}"
    ) == (0.1, 0.2, 0.3)
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_PROMPT) == 6
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION) == 1
    assert attributes.get(SpanAttributes.LLM_TOKEN_COUNT_TOTAL) == 6
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


async def test_aembedding_with_invalid_model_triggers_exception_event(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with pytest.raises(Exception):
        await litellm.aembedding(model="invalid-model-name", input=["good morning from litellm"])

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, "Expected one span to be recorded"

    span = spans[0]
    assert span.name == "CreateEmbeddings"
    assert span.status.status_code == StatusCode.ERROR

    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert (
        attributes.get(
            f"{SpanAttributes.EMBEDDING_EMBEDDINGS}.0.{EmbeddingAttributes.EMBEDDING_TEXT}"
        )
        == "good morning from litellm"
    )

    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1, "Expected one exception event to be recorded"

    exception_attributes = cast(Mapping[str, AttributeValue], exception_events[0].attributes)
    assert "exception.type" in exception_attributes
    assert "exception.message" in exception_attributes
    assert "exception.stacktrace" in exception_attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_image_generation_url(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    mock_response_image_gen = ImageResponse(
        created=1722359754,
        data=[ImageObject(b64_json=None, revised_prompt=None, url="https://dummy-url")],  # type: ignore
    )

    with patch.object(
        OpenAIChatCompletion, "image_generation", return_value=mock_response_image_gen
    ):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                litellm.image_generation(
                    model="dall-e-2",
                    prompt="a sunrise over the mountains",
                )
        else:
            litellm.image_generation(
                model="dall-e-2",
                prompt="a sunrise over the mountains",
            )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "image_generation"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "dall-e-2"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "a sunrise over the mountains"

    assert attributes.get(ImageAttributes.IMAGE_URL) == "https://dummy-url"
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "https://dummy-url"
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


@pytest.mark.parametrize("use_context_attributes", [False, True])
def test_image_generation_b64json(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    mock_response_image_gen = ImageResponse(
        created=1722359754,
        data=[ImageObject(b64_json="dummy_b64_json", revised_prompt=None, url=None)],  # type: ignore
    )

    with patch.object(
        OpenAIChatCompletion, "image_generation", return_value=mock_response_image_gen
    ):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                litellm.image_generation(
                    model="dall-e-2",
                    prompt="a sunrise over the mountains",
                )
        else:
            litellm.image_generation(
                model="dall-e-2",
                prompt="a sunrise over the mountains",
            )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "image_generation"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "dall-e-2"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "a sunrise over the mountains"

    assert attributes.get(ImageAttributes.IMAGE_URL) == "dummy_b64_json"
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "dummy_b64_json"
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


def test_image_generation_with_invalid_model_triggers_exception_event(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with pytest.raises(Exception):
        litellm.image_generation(
            model="invalid-model-name",
            prompt="a sunrise over the mountains",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, "Expected one span to be recorded"

    span = spans[0]
    assert span.name == "image_generation"
    assert span.status.status_code == StatusCode.ERROR

    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1, "Expected one exception event to be recorded"

    exception_attributes = cast(Mapping[str, AttributeValue], exception_events[0].attributes)
    assert "exception.type" in exception_attributes
    assert "exception.message" in exception_attributes
    assert "exception.stacktrace" in exception_attributes


@pytest.mark.parametrize("use_context_attributes", [False, True])
async def test_aimage_generation(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
    use_context_attributes: bool,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    in_memory_span_exporter.clear()

    mock_response_image_gen = ImageResponse(
        created=1722359754,
        data=[ImageObject(b64_json=None, revised_prompt=None, url="https://dummy-url")],  # type: ignore
    )
    with patch.object(
        OpenAIChatCompletion, "aimage_generation", return_value=mock_response_image_gen
    ):
        if use_context_attributes:
            with using_attributes(
                session_id=session_id,
                user_id=user_id,
                metadata=metadata,
                tags=tags,
                prompt_template=prompt_template,
                prompt_template_version=prompt_template_version,
                prompt_template_variables=prompt_template_variables,
            ):
                await litellm.aimage_generation(
                    model="dall-e-2",
                    prompt="a sunrise over the mountains",
                )
        else:
            await litellm.aimage_generation(
                model="dall-e-2",
                prompt="a sunrise over the mountains",
            )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "aimage_generation"
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))
    assert attributes.get(SpanAttributes.LLM_MODEL_NAME) == "dall-e-2"
    assert attributes.get(SpanAttributes.INPUT_VALUE) == "a sunrise over the mountains"

    assert attributes.get(ImageAttributes.IMAGE_URL) == "https://dummy-url"
    assert attributes.get(SpanAttributes.OUTPUT_VALUE) == "https://dummy-url"
    assert span.status.status_code == StatusCode.OK

    if use_context_attributes:
        _check_context_attributes(
            attributes,
            session_id,
            user_id,
            metadata,
            tags,
            prompt_template,
            prompt_template_version,
            prompt_template_variables,
        )


async def test_aimage_generation_with_invalid_model_triggers_exception_event(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: None,
) -> None:
    in_memory_span_exporter.clear()

    with pytest.raises(Exception):
        await litellm.aimage_generation(
            model="invalid-model-name",
            prompt="a sunrise over the mountains",
        )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1, "Expected one span to be recorded"

    span = spans[0]
    assert span.name == "aimage_generation"
    assert span.status.status_code == StatusCode.ERROR

    exception_events = [e for e in span.events if e.name == "exception"]
    assert len(exception_events) == 1, "Expected one exception event to be recorded"

    exception_attributes = cast(Mapping[str, AttributeValue], exception_events[0].attributes)
    assert "exception.type" in exception_attributes
    assert "exception.message" in exception_attributes
    assert "exception.stacktrace" in exception_attributes


def test_uninstrument(tracer_provider: TracerProvider) -> None:
    func_names = [
        "completion",
        "acompletion",
        "completion_with_retries",
        # "acompletion_with_retries",
        "embedding",
        "aembedding",
        "responses",
        "aresponses",
        "image_generation",
        "aimage_generation",
    ]

    # Instrument functions
    instrumentor = LiteLLMInstrumentor(tracer_provider=tracer_provider)
    instrumentor.instrument()

    # Check that the functions are instrumented
    for func_name in func_names:
        instrumented_func = getattr(litellm, func_name)
        assert instrumented_func.is_wrapper

    instrumentor.uninstrument()

    # Test that liteLLM functions are uninstrumented
    for func_name in func_names:
        uninstrumented_func = getattr(litellm, func_name)
        assert uninstrumented_func.__name__ == func_name

    instrumentor.instrument()

    # Check that the functions are re-instrumented
    for func_name in func_names:
        instrumented_func = getattr(litellm, func_name)
        assert instrumented_func.is_wrapper


def _check_context_attributes(
    attributes: Dict[str, Any],
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    assert attributes.pop(SpanAttributes.SESSION_ID, None) == session_id
    assert attributes.pop(SpanAttributes.USER_ID, None) == user_id
    attr_metadata = attributes.pop(SpanAttributes.METADATA, None)
    assert attr_metadata is not None
    assert isinstance(attr_metadata, str)  # must be json string
    metadata_dict = json.loads(attr_metadata)
    assert metadata_dict == metadata
    attr_tags = attributes.pop(SpanAttributes.TAG_TAGS, None)
    assert attr_tags is not None
    assert len(attr_tags) == len(tags)
    assert list(attr_tags) == tags
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE, None) == prompt_template
    assert (
        attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION, None) == prompt_template_version
    )
    assert attributes.pop(SpanAttributes.LLM_PROMPT_TEMPLATE_VARIABLES, None) == json.dumps(
        prompt_template_variables
    )


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
        "test-dict": {
            "key-1": "val-1",
            "key-2": "val-2",
        },
    }


@pytest.fixture()
def tags() -> List[str]:
    return ["tag-1", "tag-2"]


@pytest.fixture
def prompt_template() -> str:
    return (
        "This is a test prompt template with int {var_int}, "
        "string {var_string}, and list {var_list}"
    )


@pytest.fixture
def prompt_template_version() -> str:
    return "v1.0"


@pytest.fixture
def prompt_template_variables() -> Dict[str, Any]:
    return {
        "var_int": 1,
        "var_str": "2",
        "var_list": [1, 2, 3],
    }


def _check_llm_message(
    prefix: str,
    i: int,
    attributes: Dict[str, Any],
    message: Dict[str, Any],
    hide_text: bool = False,
    hide_images: bool = False,
    image_limit: Optional[int] = None,
) -> None:
    assert attributes.pop(message_role(prefix, i), None) == message.get("role")
    expected_content = message.get("content")
    if isinstance(expected_content, list):
        for j, expected_content_item in enumerate(expected_content):
            content_item_type = attributes.pop(message_contents_type(prefix, i, j), None)
            expected_content_item_type = expected_content_item.get("type")
            if expected_content_item_type == "image_url":
                expected_content_item_type = "image"
            assert content_item_type == expected_content_item_type
            if content_item_type == "text":
                content_item_text = attributes.pop(message_contents_text(prefix, i, j), None)
                if hide_text:
                    assert content_item_text == REDACTED_VALUE
                else:
                    assert content_item_text == expected_content_item.get("text")
            elif content_item_type == "image":
                content_item_image_url = attributes.pop(
                    message_contents_image_url(prefix, i, j), None
                )
                if hide_images:
                    assert content_item_image_url is None
                else:
                    expected_url = expected_content_item.get("image_url").get("url")
                    if image_limit is not None and len(expected_url) > image_limit:
                        assert content_item_image_url == REDACTED_VALUE
                    else:
                        assert content_item_image_url == expected_url
    else:
        content = attributes.pop(message_content(prefix, i), None)
        if expected_content is not None and hide_text:
            assert content == REDACTED_VALUE
        else:
            assert content == expected_content


def message_content(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENT}"


def message_role(prefix: str, i: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_ROLE}"


def message_contents_type(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TYPE}"


def message_contents_text(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_TEXT}"


def message_contents_image_url(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MESSAGE_CONTENT_IMAGE}.{IMAGE_URL}"


@pytest.mark.parametrize(
    "model_name,expected_provider",
    [
        pytest.param("gpt-4o", "openai", id="openai"),
        pytest.param("claude-3-haiku-20240307", "anthropic", id="anthropic"),
        pytest.param("azure/gpt-4", "azure", id="azure"),
        pytest.param("bedrock/anthropic.claude-3-sonnet-20240229-v1:0", "aws", id="aws"),
        pytest.param("vertex_ai/gemini-1.5-pro", "google", id="google"),
        pytest.param("cohere/command", "cohere", id="cohere"),
        pytest.param("mistral/mistral-medium", "mistralai", id="mistralai"),
        pytest.param("xai/grok-beta", "xai", id="xai"),
        pytest.param("deepseek/deepseek-chat", "deepseek", id="deepseek"),
        pytest.param("huggingface/together/deepseek-ai/DeepSeek-R1", None, id="unknown-provider"),
    ],
)
def test_provider_attribute_correctly_set(
    model_name: str,
    expected_provider: Optional[str],
    setup_litellm_instrumentation: Any,
    in_memory_span_exporter: InMemorySpanExporter,
) -> None:
    litellm.completion(
        model=model_name,
        messages=[{"content": "Hello", "role": "user"}],
        mock_response="Hi there!",
    )

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = span.attributes
    assert attributes is not None
    provider = attributes.get(SpanAttributes.LLM_PROVIDER)
    assert provider == expected_provider


MESSAGE_CONTENT = MessageAttributes.MESSAGE_CONTENT
MESSAGE_ROLE = MessageAttributes.MESSAGE_ROLE
MESSAGE_CONTENTS = MessageAttributes.MESSAGE_CONTENTS
MESSAGE_CONTENT_TYPE = MessageContentAttributes.MESSAGE_CONTENT_TYPE
MESSAGE_CONTENT_IMAGE = MessageContentAttributes.MESSAGE_CONTENT_IMAGE
MESSAGE_CONTENT_TEXT = MessageContentAttributes.MESSAGE_CONTENT_TEXT
IMAGE_URL = ImageAttributes.IMAGE_URL
REDACTED_VALUE = "__REDACTED__"
