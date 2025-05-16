import os

import phoenix as px
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from openinference.instrumentation.pydanticai import (
    OpenInferenceSpanExporter,
)

# Get the secret key from environment variables
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Launch Phoenix app
session = px.launch_app()

tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)

endpoint = "http://127.0.0.1:6006/v1/traces"
exporter = OTLPSpanExporter(endpoint=endpoint)

# Use the OpenInferenceSpanExporter to capture OpenInference spans from Pydantic AI
openInferenceExporter = OpenInferenceSpanExporter(exporter)
tracer_provider.add_span_processor(SimpleSpanProcessor(openInferenceExporter))


class LocationModel(BaseModel):
    city: str
    country: str


model = OpenAIModel("gpt-4o", provider=OpenAIProvider())

agent = Agent(model, output_type=LocationModel, instrument=True)

if __name__ == "__main__":
    result = agent.run_sync("The windy city in the US of A.")
    print(result)
