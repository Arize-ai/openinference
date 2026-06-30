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


def test_completion_with_reasoning_content(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    from litellm.types.utils import Choices, ModelResponse

    in_memory_span_exporter.clear()

    msg = LitellmMessage.model_construct(
        role="assistant",
        content="Beijing",
        reasoning_content="I need to think about this.",
    )
    response = ModelResponse(
        id="chatcmpl-rc1",
        choices=[Choices(message=msg, index=0, finish_reason="stop")],
        model="deepseek/deepseek-reasoner",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = lambda *args, **kwargs: response
        litellm.completion(
            model="deepseek/deepseek-reasoner",
            messages=[{"role": "user", "content": "What is the capital of China?"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES

    # Block 0 is the reasoning block
    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_text(prefix, 0, 0)) == "I need to think about this."
    # Block 1 is the text block
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "Beijing"
    # Flat message.content must NOT be set when structured blocks are present
    assert attributes.get(f"{prefix}.0.{MESSAGE_CONTENT}") is None


def test_completion_streaming_with_reasoning_content(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Streaming: delta.reasoning_content chunks accumulate and are emitted as a reasoning block."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    in_memory_span_exporter.clear()

    chunks = [
        ModelResponseStream(
            id="chatcmpl-rs1",
            model="deepseek/deepseek-reasoner",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role="assistant", content=None, reasoning_content="thinking "
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-rs1",
            model="deepseek/deepseek-reasoner",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role=None, content="answer", reasoning_content="hard"
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-rs1",
            model="deepseek/deepseek-reasoner",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(role=None, content=None, reasoning_content=None),
                    finish_reason="stop",
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

    input_messages = [{"content": "What is the capital of China?", "role": "user"}]
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
                model="deepseek/deepseek-reasoner",
                messages=input_messages,
                stream=True,
            )
            chunks_received = list(response)
            assert len(chunks_received) == 3
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES

    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_text(prefix, 0, 0)) == "thinking hard"
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "answer"


@pytest.mark.asyncio
async def test_acompletion_streaming_with_reasoning_content(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    in_memory_span_exporter.clear()

    chunks = [
        ModelResponseStream(
            id="chatcmpl-ars1",
            model="deepseek/deepseek-reasoner",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role="assistant", content=None, reasoning_content="thinking "
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-ars1",
            model="deepseek/deepseek-reasoner",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role=None, content="answer", reasoning_content="hard"
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-ars1",
            model="deepseek/deepseek-reasoner",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(role=None, content=None, reasoning_content=None),
                    finish_reason="stop",
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

    input_messages = [{"content": "What is the capital of China?", "role": "user"}]
    original_func = LiteLLMInstrumentor.original_litellm_funcs["acompletion"]

    async def mock_acompletion(*args: Any, **kwargs: Any) -> MockAsyncStreamWrapper:
        return MockAsyncStreamWrapper(chunks)

    try:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = mock_acompletion
        response = await litellm.acompletion(
            model="deepseek/deepseek-reasoner",
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
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES

    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_text(prefix, 0, 0)) == "thinking hard"
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "answer"


def test_completion_with_thinking_blocks_and_signature(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    from litellm.types.utils import Choices, ModelResponse

    in_memory_span_exporter.clear()

    msg = LitellmMessage.model_construct(
        role="assistant",
        content="result",
        thinking_blocks=[{"type": "thinking", "thinking": "step 1", "signature": "sig_abc"}],
    )
    response = ModelResponse(
        id="chatcmpl-tb1",
        choices=[Choices(message=msg, index=0, finish_reason="stop")],
        model="claude-3-7-sonnet",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = lambda *args, **kwargs: response
        litellm.completion(
            model="claude-3-7-sonnet",
            messages=[{"role": "user", "content": "Solve step by step"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
    sig_key = (
        f"{prefix}.0.{MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}"
    )

    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_text(prefix, 0, 0)) == "step 1"
    assert attributes.get(sig_key) == "sig_abc"
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "result"


def test_completion_with_inline_thinking_content_blocks(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Non-streaming: content list with type='thinking' dicts are captured as reasoning blocks."""
    from litellm.types.utils import Choices, ModelResponse

    in_memory_span_exporter.clear()

    msg = LitellmMessage.model_construct(
        role="assistant",
        content=[
            {"type": "thinking", "thinking": "reasoning text", "signature": "sig_xyz"},
            {"type": "text", "text": "answer"},
        ],
    )
    response = ModelResponse(
        id="chatcmpl-it1",
        choices=[Choices(message=msg, index=0, finish_reason="stop")],
        model="claude-3-7-sonnet",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = lambda *args, **kwargs: response
        litellm.completion(
            model="claude-3-7-sonnet",
            messages=[{"role": "user", "content": "Think and answer"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
    sig_key = (
        f"{prefix}.0.{MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}"
    )

    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_text(prefix, 0, 0)) == "reasoning text"
    assert attributes.get(sig_key) == "sig_xyz"
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "answer"


def test_completion_content_id_not_emitted(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """'id' field inside a thinking content block must never appear as a span attribute."""
    from litellm.types.utils import Choices, ModelResponse

    in_memory_span_exporter.clear()

    msg = LitellmMessage.model_construct(
        role="assistant",
        content=[
            {"type": "thinking", "thinking": "r", "signature": "s", "id": "do_not_emit_me"},
        ],
    )
    response = ModelResponse(
        id="chatcmpl-noid1",
        choices=[Choices(message=msg, index=0, finish_reason="stop")],
        model="claude-3-7-sonnet",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = lambda *args, **kwargs: response
        litellm.completion(
            model="claude-3-7-sonnet",
            messages=[{"role": "user", "content": "Think"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_text(prefix, 0, 0)) == "r"
    for key in attributes:
        assert "message_content.id" not in key, f"Unexpected id attribute found: {key}"


def test_completion_with_thinking_blocks_and_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    from litellm.types.utils import Choices, ModelResponse

    in_memory_span_exporter.clear()

    msg = LitellmMessage.model_construct(
        role="assistant",
        content="ok",
        thinking_blocks=[{"type": "thinking", "thinking": "let me reason"}],
        tool_calls=[{"function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}],
    )
    response = ModelResponse(
        id="chatcmpl-tbtc1",
        choices=[Choices(message=msg, index=0, finish_reason="tool_calls")],
        model="claude-3-7-sonnet",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = lambda *args, **kwargs: response
        litellm.completion(
            model="claude-3-7-sonnet",
            messages=[{"role": "user", "content": "Get weather for Paris"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES

    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_text(prefix, 0, 0)) == "let me reason"
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "ok"

    tool_call_name_key = (
        f"{prefix}.0.{MessageAttributes.MESSAGE_TOOL_CALLS}.0."
        f"{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"
    )
    assert attributes.get(tool_call_name_key) == "get_weather"


def test_completion_with_redacted_thinking_block(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Non-streaming: a single redacted_thinking block in thinking_blocks is emitted as
    a reasoning block with data, followed by the text content block."""
    from litellm.types.utils import Choices, ModelResponse

    in_memory_span_exporter.clear()

    msg = LitellmMessage.model_construct(
        role="assistant",
        content="result",
        thinking_blocks=[{"type": "redacted_thinking", "data": "enc_blob_abc"}],
    )
    response = ModelResponse(
        id="chatcmpl-rt1",
        choices=[Choices(message=msg, index=0, finish_reason="stop")],
        model="claude-3-7-sonnet",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = lambda *args, **kwargs: response
        litellm.completion(
            model="claude-3-7-sonnet",
            messages=[{"role": "user", "content": "Think step by step"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES

    # Block 0 is the redacted reasoning block
    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_data(prefix, 0, 0)) == "enc_blob_abc"
    # No text field on a redacted block
    assert attributes.get(message_contents_text(prefix, 0, 0)) is None
    # Block 1 is the text content
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "result"


def test_completion_with_mixed_thinking_and_redacted_thinking(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Non-streaming: thinking_blocks with both a visible thinking block and a redacted_thinking
    block are each emitted as reasoning blocks in order, followed by the text block."""
    from litellm.types.utils import Choices, ModelResponse

    in_memory_span_exporter.clear()

    msg = LitellmMessage.model_construct(
        role="assistant",
        content="answer",
        thinking_blocks=[
            {"type": "thinking", "thinking": "step1", "signature": "sig1"},
            {"type": "redacted_thinking", "data": "enc_data"},
        ],
    )
    response = ModelResponse(
        id="chatcmpl-mrt1",
        choices=[Choices(message=msg, index=0, finish_reason="stop")],
        model="claude-3-7-sonnet",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = lambda *args, **kwargs: response
        litellm.completion(
            model="claude-3-7-sonnet",
            messages=[{"role": "user", "content": "Think and answer"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES
    sig_key = (
        f"{prefix}.0.{MESSAGE_CONTENTS}.0.{MessageContentAttributes.MESSAGE_CONTENT_SIGNATURE}"
    )

    # Block 0: visible thinking block
    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_text(prefix, 0, 0)) == "step1"
    assert attributes.get(sig_key) == "sig1"
    # Block 1: redacted thinking block
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "reasoning"
    assert attributes.get(message_contents_data(prefix, 0, 1)) == "enc_data"
    assert attributes.get(message_contents_text(prefix, 0, 1)) is None
    # Block 2: text content
    assert attributes.get(message_contents_type(prefix, 0, 2)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 2)) == "answer"


def test_completion_streaming_with_redacted_thinking_block(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Streaming: a delta.thinking_blocks entry of type redacted_thinking is accumulated and
    emitted as a reasoning block with data, followed by the text content block."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    in_memory_span_exporter.clear()

    chunks = [
        ModelResponseStream(
            id="chatcmpl-rts1",
            model="claude-3-7-sonnet",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role="assistant",
                        content=None,
                        reasoning_content=None,
                        thinking_blocks=[
                            {"type": "redacted_thinking", "data": "enc_streaming_blob"}
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-rts1",
            model="claude-3-7-sonnet",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role=None,
                        content="streamed answer",
                        reasoning_content=None,
                        thinking_blocks=None,
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-rts1",
            model="claude-3-7-sonnet",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role=None, content=None, reasoning_content=None, thinking_blocks=None
                    ),
                    finish_reason="stop",
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

    input_messages = [{"content": "Think about this", "role": "user"}]
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
                model="claude-3-7-sonnet",
                messages=input_messages,
                stream=True,
            )
            chunks_received = list(response)
            assert len(chunks_received) == 3
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES

    # Block 0: redacted reasoning block collected from streaming
    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_data(prefix, 0, 0)) == "enc_streaming_blob"
    assert attributes.get(message_contents_text(prefix, 0, 0)) is None
    # Block 1: text content from streaming
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "streamed answer"


@pytest.mark.asyncio
async def test_acompletion_streaming_with_redacted_thinking_block(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Async streaming: delta.thinking_blocks with redacted_thinking are captured correctly."""
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    in_memory_span_exporter.clear()

    chunks = [
        ModelResponseStream(
            id="chatcmpl-arts1",
            model="claude-3-7-sonnet",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role="assistant",
                        content=None,
                        reasoning_content=None,
                        thinking_blocks=[{"type": "redacted_thinking", "data": "enc_async_blob"}],
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-arts1",
            model="claude-3-7-sonnet",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role=None,
                        content="async answer",
                        reasoning_content=None,
                        thinking_blocks=None,
                    ),
                    finish_reason=None,
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-arts1",
            model="claude-3-7-sonnet",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta.model_construct(
                        role=None, content=None, reasoning_content=None, thinking_blocks=None
                    ),
                    finish_reason="stop",
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

    input_messages = [{"content": "Think about this", "role": "user"}]
    original_func = LiteLLMInstrumentor.original_litellm_funcs["acompletion"]

    async def mock_acompletion(*args: Any, **kwargs: Any) -> MockAsyncStreamWrapper:
        return MockAsyncStreamWrapper(chunks)

    try:
        LiteLLMInstrumentor.original_litellm_funcs["acompletion"] = mock_acompletion
        response = await litellm.acompletion(
            model="claude-3-7-sonnet",
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
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES

    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_data(prefix, 0, 0)) == "enc_async_blob"
    assert attributes.get(message_contents_text(prefix, 0, 0)) is None
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "async answer"


def test_completion_with_inline_redacted_thinking_content_block(
    in_memory_span_exporter: InMemorySpanExporter,
    setup_litellm_instrumentation: Any,
) -> None:
    """Non-streaming: content list containing type='redacted_thinking' dict exercises the
    _get_attributes_from_message_content() path and emits a reasoning block with data."""
    from litellm.types.utils import Choices, ModelResponse

    in_memory_span_exporter.clear()

    msg = LitellmMessage.model_construct(
        role="assistant",
        content=[
            {"type": "redacted_thinking", "data": "enc_inline_blob"},
            {"type": "text", "text": "reply"},
        ],
    )
    response = ModelResponse(
        id="chatcmpl-irt1",
        choices=[Choices(message=msg, index=0, finish_reason="stop")],
        model="claude-3-7-sonnet",
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )

    original_func = LiteLLMInstrumentor.original_litellm_funcs["completion"]
    try:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = lambda *args, **kwargs: response
        litellm.completion(
            model="claude-3-7-sonnet",
            messages=[{"role": "user", "content": "Think and reply"}],
        )
    finally:
        LiteLLMInstrumentor.original_litellm_funcs["completion"] = original_func

    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(cast(Mapping[str, AttributeValue], span.attributes))

    prefix = SpanAttributes.LLM_OUTPUT_MESSAGES

    # Block 0: inline redacted_thinking content block
    assert attributes.get(message_contents_type(prefix, 0, 0)) == "reasoning"
    assert attributes.get(message_contents_data(prefix, 0, 0)) == "enc_inline_blob"
    # Block 1: text content block
    assert attributes.get(message_contents_type(prefix, 0, 1)) == "text"
    assert attributes.get(message_contents_text(prefix, 0, 1)) == "reply"


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


def message_contents_data(prefix: str, i: int, j: int) -> str:
    return f"{prefix}.{i}.{MESSAGE_CONTENTS}.{j}.{MessageContentAttributes.MESSAGE_CONTENT_DATA}"


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
