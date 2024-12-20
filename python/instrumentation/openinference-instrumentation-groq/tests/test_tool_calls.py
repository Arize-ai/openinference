import json
from typing import Any, Dict, List, Optional, Type, Union, cast

import pytest
from groq import Groq
from groq._base_client import _StreamT
from groq._types import Body, RequestFiles, RequestOptions, ResponseT
from groq.types import CompletionUsage
from groq.types.chat import ChatCompletion, ChatCompletionMessage, ChatCompletionToolParam
from groq.types.chat.chat_completion import Choice
from groq.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def _mock_post(
    self: Any,
    path: str = "fake/url",
    *,
    cast_to: Type[ResponseT],
    body: Optional[Body] = None,
    options: RequestOptions = {},
    files: Optional[RequestFiles] = None,
    stream: bool = False,
    stream_cls: Optional[Type[_StreamT]] = None,
) -> Union[ResponseT, _StreamT]:
    """
    opts = FinalRequestOptions.construct(
        method="post", url=path, json_data=body, files=to_httpx_files(files), **options
    )
    return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))
    """
    mock_tool_completion = ChatCompletion(
        id="chat_comp_0",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="",
                    role="assistant",
                    function_call=None,
                    tool_calls=[
                        ChatCompletionMessageToolCall(
                            id="call_62136357",
                            function=Function(
                                arguments='{"city": "San Francisco"}', name="get_weather"
                            ),
                            type="function",
                        ),
                        ChatCompletionMessageToolCall(
                            id="call_62136358",
                            function=Function(
                                arguments='{"city": "San Francisco"}', name="get_population"
                            ),
                            type="function",
                        ),
                    ],
                ),
            )
        ],
        created=1722531851,
        model="test_groq_model",
        object="chat.completion",
        system_fingerprint="fp0",
        usage=CompletionUsage(
            completion_tokens=379,
            prompt_tokens=25,
            total_tokens=404,
            completion_time=0.616262398,
            prompt_time=0.002549632,
            queue_time=None,
            total_time=0.6188120300000001,
        ),
    )
    return cast(ResponseT, mock_tool_completion)


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


def test_tool_calls(
    tracer_provider: TracerProvider,
    in_memory_span_exporter: InMemorySpanExporter,
    setup_groq_instrumentation: Any,
    session_id: str,
    user_id: str,
    metadata: Dict[str, Any],
    tags: List[str],
    prompt_template: str,
    prompt_template_version: str,
    prompt_template_variables: Dict[str, Any],
) -> None:
    client = Groq(api_key="fake-api-key")
    client.chat.completions._post = _mock_post  # type: ignore[assignment]

    input_tools = [
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_weather",
                "description": "finds the weather for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for, e.g. 'London'",
                        }
                    },
                    "required": ["city"],
                },
            },
        ),
        ChatCompletionToolParam(
            type="function",
            function={
                "name": "get_population",
                "description": "finds the population for a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the population for, e.g. 'London'",
                        }
                    },
                    "required": ["city"],
                },
            },
        ),
    ]

    client.chat.completions.create(
        model="test_groq_model",
        tools=input_tools,
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_62136355",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "New York"}',
                        },
                    },
                    {
                        "id": "call_62136356",
                        "type": "function",
                        "function": {
                            "name": "get_population",
                            "arguments": '{"city": "New York"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_62136355",
                "content": '{"city": "New York", "weather": "fine"}',
            },
            {
                "role": "tool",
                "tool_call_id": "call_62136356",
                "content": '{"city": "New York", "weather": "large"}',
            },
            {
                "role": "assistant",
                "content": "In New York the weather is fine and the population is large.",
            },
            {
                "role": "user",
                "content": "What's the weather and population in San Francisco?",
            },
        ],
    )
    spans = in_memory_span_exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attributes = dict(span.attributes or {})
    for i in range(len(input_tools)):
        json_schema = attributes.pop(f"llm.tools.{i}.tool.json_schema")
        assert isinstance(json_schema, str)
        assert json.loads(json_schema)
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.0.tool_call.id") == "call_62136355"
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.0.tool_call.function.name")
        == "get_weather"
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.0.tool_call.function.arguments")
        == '{"city": "New York"}'
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.1.tool_call.id") == "call_62136356"
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.1.tool_call.function.name")
        == "get_population"
    )
    assert (
        attributes.pop("llm.input_messages.0.message.tool_calls.1.tool_call.function.arguments")
        == '{"city": "New York"}'
    )
    assert attributes.pop("llm.input_messages.1.message.role") == "tool"
    assert attributes.pop("llm.input_messages.1.message.tool_call_id") == "call_62136355"
    assert (
        attributes.pop("llm.input_messages.1.message.content")
        == '{"city": "New York", "weather": "fine"}'
    )
    assert attributes.pop("llm.input_messages.2.message.role") == "tool"
    assert attributes.pop("llm.input_messages.2.message.tool_call_id") == "call_62136356"
    assert (
        attributes.pop("llm.input_messages.2.message.content")
        == '{"city": "New York", "weather": "large"}'
    )
    assert attributes.pop("llm.output_messages.0.message.tool_calls.0.tool_call.id")
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.0.tool_call.function.name")
        == "get_weather"
    )
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.0.tool_call.function.arguments")
        == '{"city": "San Francisco"}'
    )
    assert attributes.pop("llm.output_messages.0.message.tool_calls.1.tool_call.id")
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.1.tool_call.function.name")
        == "get_population"
    )
    assert (
        attributes.pop("llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments")
        == '{"city": "San Francisco"}'
    )
