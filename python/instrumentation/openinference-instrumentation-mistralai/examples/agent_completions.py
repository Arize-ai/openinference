import os

from mistralai.client import Mistral
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.mistralai import MistralAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

MistralAIInstrumentor().instrument(tracer_provider=tracer_provider)

if __name__ == "__main__":
    client = Mistral(
        api_key=os.getenv("MISTRAL_API_KEY", ""),
    )
    response = client.agents.complete(
        agent_id="your-agent-id",
        messages=[
            {"role": "user", "content": "plan a vacation for me in Tbilisi"},
        ],
    )
    print(response)
