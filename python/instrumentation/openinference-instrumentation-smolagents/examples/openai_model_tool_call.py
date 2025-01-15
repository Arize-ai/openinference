import os

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
)
from smolagents import OpenAIServerModel
from smolagents.tools import Tool

from openinference.instrumentation.smolagents import SmolagentsInstrumentor

endpoint = "http://0.0.0.0:6006/v1/traces"
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider, skip_dep_check=True)


class GetWeatherTool(Tool):
    name = "get_weather"
    description = "Get the weather for a given city"
    inputs = {"location": {"type": "string", "description": "The city to get the weather for"}}
    output_type = "string"

    def forward(self, location: str) -> str:
        return "sunny"


model = OpenAIServerModel(
    model_id="gpt-4o", api_key=os.environ["OPENAI_API_KEY"], api_base="https://api.openai.com/v1"
)
output_message = model(
    messages=[
        {
            "role": "user",
            "content": "What is the weather in Paris?",
        }
    ],
    tools_to_call_from=[GetWeatherTool()],
)
print(output_message)
