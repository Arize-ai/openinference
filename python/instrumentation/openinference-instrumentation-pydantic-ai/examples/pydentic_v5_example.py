from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from pydantic_ai import Agent, InstrumentationSettings

from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor

endpoint = "http://localhost:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
exporter = OTLPSpanExporter(endpoint=endpoint)
trace.set_tracer_provider(tracer_provider)
tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

agent = Agent("openai:gpt-4.1-nano")
agent.instrument_all(instrument=InstrumentationSettings(version=2))


@agent.tool_plain
def get_weather(city: str) -> str:
    return f"It's sunny in {city}."


if __name__ == "__main__":
    response = agent.run_sync("What's the weather in Paris?")
    print(response)
