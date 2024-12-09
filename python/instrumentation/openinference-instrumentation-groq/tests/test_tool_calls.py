import json
from typing import Any, Optional, Type, Union, cast

import pytest
from groq import Groq
from groq._base_client import _StreamT
from groq._types import Body, RequestFiles, RequestOptions, ResponseT
from groq.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionToolParam,
)
from groq.types.chat.chat_completion import Choice, CompletionUsage
from groq.types.chat.chat_completion_message_tool_call import Function
from opentelemetry import trace as trace_api
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter


def create_mock_tool_completion(messages):
    last_user_message = next(msg for msg in messages[::-1] if msg.get("role") == "user")
    city = last_user_message["content"].split(" in ")[-1].split("?")[0].strip()

    # Create tool calls with dynamically generated IDs
    tool_calls = [
        ChatCompletionMessageToolCall(
            id=f"call_{62136355 + i}",  # Use a base ID and increment
            function=Function(arguments=json.dumps({"city": city}), name=tool_name),
            type="function",
        )
        for i, tool_name in enumerate(["get_weather", "get_population"])
    ]

    return ChatCompletion(
        id="chat_comp_0",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                logprobs=None,
                message=ChatCompletionMessage(
                    content="", role="assistant", function_call=None, tool_calls=tool_calls
                ),
            )
        ],
        created=1722531851,
        model="fake_groq_model",
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
    # Extract messages from the request body
    messages = body.get("messages", []) if body else []

    # Create a mock completion based on the messages
    mock_completion = create_mock_tool_completion(messages)

    return cast(ResponseT, mock_completion)


@pytest.fixture()
def in_memory_span_exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture()
def tracer_provider(in_memory_span_exporter: InMemorySpanExporter) -> TracerProvider:
    resource = Resource(attributes={})
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_span_exporter))
    return tracer_provider


@pytest.mark.vcr(
    decode_compressed_response=True,
    before_record_request=lambda _: _.headers.clear() or _,
    before_record_response=lambda _: {**_, "headers": {}},
)
def test_tool_calls(
    in_memory_span_exporter: InMemorySpanExporter,
    tracer_provider: trace_api.TracerProvider,
) -> None:
    client = Groq(api_key="fake")
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
        extra_headers={"Accept-Encoding": "gzip"},
        model="fake_groq_model",
        tools=input_tools,
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_62136355",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "New York"}'},
                    },
                    {
                        "id": "call_62136356",
                        "type": "function",
                        "function": {"name": "get_population", "arguments": '{"city": "New York"}'},
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
