import asyncio
import os

from google import genai
from google.genai.types import Content, GenerateContentConfig, Part
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,  # type: ignore[import-not-found]
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = genai.Client()


def get_weather(location: str):
    """Gets the weather for a given location."""
    return f"The weather in {location} is sunny."


weather_tool = {
    "type": "function",
    "name": "get_weather",
    "description": "Gets the weather for a given location.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
        },
        "required": ["location"]
    }
}


def run():
    interaction = client.interactions.create(
        model="gemini-3-flash-preview",
        input="What is the weather in Paris & New Delhi?",
        tools=[weather_tool]
    )
    # for output in interaction.outputs:
    #     if output.type == "function_call":
    #         print(f"Tool Call: {output.name}({output.arguments})")
    #         # Execute tool
    #         result = get_weather(**output.arguments)
    #         print(f"Result: {result}")
    #
    #         # Send result back
    #         interaction = client.interactions.create(
    #             model="gemini-3-flash-preview",
    #             previous_interaction_id=interaction.id,
    #             input=[{
    #                 "type": "function_result",
    #                 "name": output.name,
    #                 "call_id": output.id,
    #                 "result": result
    #             }]
    #         )
    #         print(f"Response: {interaction.outputs[-1].text}")


if __name__ == "__main__":
    run()

