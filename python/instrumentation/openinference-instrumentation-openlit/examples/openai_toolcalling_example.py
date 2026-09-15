"""
Tool-Calling OpenAI Chat with OpenLIT → Phoenix Observability

This script shows how to:
- Instrument OpenAI tool-calling requests with OpenTelemetry.
- Convert traces to OpenInference format using OpenInferenceSpanProcessor.
- Export spans to Phoenix via OTLP gRPC.
- Define tools (e.g., weather, traffic) and serve tool outputs.
"""

import json
import os
import sys

import grpc
import openai
import openlit
from dotenv import load_dotenv
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from phoenix.otel import register

from openinference.instrumentation.openlit import OpenInferenceSpanProcessor

# --------------------------------------------------------------------------------
# Load environment variables
# --------------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --------------------------------------------------------------------------------
# Debug processor (optional)
# --------------------------------------------------------------------------------
class DebugPrintProcessor(SpanProcessor):
    def on_end(self, span: ReadableSpan) -> None:
        if "gen_ai.request.model" not in span.attributes:
            return
        print(f"\n=== RAW OpenLIT span: {span.name} ===", file=sys.stderr)
        print(json.dumps(dict(span.attributes), default=str, indent=2), file=sys.stderr)


# --------------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # ------------------------------
    # Register tracer
    # ------------------------------
    provider = register(
        project_name="openai-tool-calling-demo",
        set_global_tracer_provider=True,
    )

    provider.add_span_processor(DebugPrintProcessor())
    provider.add_span_processor(OpenInferenceSpanProcessor())
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint="http://localhost:4317",
                headers={},
                compression=grpc.Compression.Gzip,
            )
        )
    )

    tracer = provider.get_tracer(__name__)
    openlit.init(otel_tracer=tracer)

    # ------------------------------
    # Functions
    # ------------------------------
    def get_weather(location: str) -> str:
        return "100 degrees"

    def get_traffic(location: str) -> str:
        return "high level traffic"

    # ------------------------------
    # Setup OpenAI client + tools
    # ------------------------------
    client = openai.OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current temperature for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_traffic",
                "description": "Get current traffic for a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia",
                        }
                    },
                    "required": ["location"],
                },
            },
        },
    ]

    # ------------------------------
    # Initial request
    # ------------------------------
    messages = [{"role": "user", "content": "What is the weather like in Paris today?"}]

    response = client.chat.completions.create(model="gpt-4.1", messages=messages, tools=tools)

    tool_registry = {
        "get_weather": get_weather,
        "get_traffic": get_traffic,
    }

    choice = response.choices[0]
    if choice.finish_reason == "tool_calls":
        if choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            function_name = tool_call.function.name

            args = json.loads(tool_call.function.arguments)
            location = args["location"]

            if function_name in tool_registry:
                tool_output = tool_registry[function_name](**args)
            else:
                raise ValueError(f"Tool function '{function_name}' not found.")

            messages.extend(
                [
                    choice.message.model_dump(),
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_output,
                    },
                ]
            )

            # Final step: provide answer after tool output
            final = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
            )
            print("Assistant:", final.choices[0].message.content)

    else:
        print("Assistant:", choice.message.content)
