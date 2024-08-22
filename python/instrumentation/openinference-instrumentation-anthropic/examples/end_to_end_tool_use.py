from typing import Optional

import anthropic
from anthropic.types import (
    Message,
    MessageParam,
    TextBlock,
    TextBlockParam,
    ToolResultBlockParam,
    ToolUseBlock,
    ToolUseBlockParam,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from typing_extensions import assert_never

from openinference.instrumentation.anthropic import AnthropicInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)


def _to_assistant_message_param(
    message: Message,
) -> MessageParam:
    content = []
    for block in message.content:
        if isinstance(block, TextBlock):
            content.append(TextBlockParam(text=block.text, type=block.type))
        elif isinstance(block, ToolUseBlock):
            content.append(
                ToolUseBlockParam(id=block.id, input=block.input, name=block.name, type=block.type)
            )
        else:
            assert_never(block)
    return MessageParam(content=content, role="assistant")


def _get_tool_use_id(message: Message) -> Optional[str]:
    for block in message.content:
        if isinstance(block, ToolUseBlock):
            return block.id
    return None


client = anthropic.Anthropic()
messages = [{"role": "user", "content": "What is the weather like in San Francisco in Fahrenheit?"}]
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
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
                        "description": 'The unit of temperature, either "celsius" or "fahrenheit"',
                    },
                },
                "required": ["location"],
            },
        }
    ],
    messages=messages,
)
messages.append(_to_assistant_message_param(response))
assert (tool_use_id := _get_tool_use_id(response)) is not None, "tool was not called"
messages.append(
    MessageParam(
        content=[
            ToolResultBlockParam(
                tool_use_id=tool_use_id,
                content='{"weather": "sunny", "temperature": "75"}',
                type="tool_result",
                is_error=False,
            )
        ],
        role="user",
    )
)
response = client.messages.create(
    model="claude-3-5-sonnet-20240620",
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
                        "description": 'The unit of temperature, either "celsius" or "fahrenheit"',
                    },
                },
                "required": ["location"],
            },
        }
    ],
    messages=messages,
)
print(f"{response=}")
