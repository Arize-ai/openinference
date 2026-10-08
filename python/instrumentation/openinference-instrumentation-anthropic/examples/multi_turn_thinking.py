from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    RedactedThinkingBlock,
    TextBlock,
    ThinkingBlock,
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


def _to_assistant_message_param(message: Message) -> MessageParam:
    content = []
    for block in message.content:
        if isinstance(block, (TextBlock, ThinkingBlock, RedactedThinkingBlock)):
            content.append(block)
        else:
            assert_never(block)
    return MessageParam(content=content, role="assistant")


client = Anthropic()
messages: list[MessageParam] = [
    {
        "role": "user",
        "content": "What is 27 * 453? Think it through step by step.",
    }
]

response = client.messages.create(
    max_tokens=2048,
    thinking={"type": "enabled", "budget_tokens": 1024},
    messages=messages,
    model="claude-sonnet-4-6",
)
print(response)

# Thinking blocks (including their signatures) must be passed back unmodified
# in the conversation history for multi-turn requests with extended thinking.
messages.append(_to_assistant_message_param(response))
messages.append(
    {
        "role": "user",
        "content": "Now divide that result by 9 and explain your reasoning.",
    }
)

response = client.messages.create(
    max_tokens=2048,
    thinking={"type": "enabled", "budget_tokens": 1024},
    messages=messages,
    model="claude-sonnet-4-6",
)
print(response)
