import random

from anthropic import Anthropic
from anthropic.types.message_param import MessageParam
from anthropic.types.model_param import ModelParam
from anthropic.types.tool_param import ToolParam
from dotenv import load_dotenv
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.anthropic import AnthropicInstrumentor

load_dotenv()

endpoint = "http://127.0.0.1:4317"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

client = Anthropic()
MODEL: ModelParam = "claude-3-5-haiku-latest"

weather_bot_prompt = """
I am a helpful weather bot. Provide me with a location and units,
and I will output the current weather.
"""
question = "what is the weather in california in imperial units?"

messages: list[MessageParam] = [
    {"role": "assistant", "content": weather_bot_prompt},
    {"role": "user", "content": question},
]

# Define the tool as a JSON schema string for Claude
tools: list[ToolParam] = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given location and units",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "units": {"type": "string", "enum": ["metric", "imperial"]},
            },
            "required": ["location", "units"],
        },
    }
]


message = client.messages.create(
    model=MODEL,
    messages=messages,
    tools=tools,
    max_tokens=1024,
)


def get_weather(location: str, units: str) -> str:
    return f"The weather in {location} is {random.randint(0, 100)} degrees {units}"


assert message.stop_reason == "tool_use"

tool = next(tool for tool in message.content if tool.type == "tool_use")
result: str = get_weather(
    location=tool.input["location"],  # type: ignore
    units=tool.input["units"],  # type: ignore
)

tool_use_message: MessageParam = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": tool.id,
            "content": [{"type": "text", "text": result}],
        },
    ],
}

new_messages: list[MessageParam] = messages + [
    {"role": message.role, "content": message.content},
    tool_use_message,
]

response = client.messages.create(
    model=MODEL,
    messages=new_messages,
    max_tokens=1024,
    tools=tools,
)

res_content = response.content[0]

if res_content.type == "text":
    print(res_content.text)

print(f"\n\nMessages used: {new_messages}")
