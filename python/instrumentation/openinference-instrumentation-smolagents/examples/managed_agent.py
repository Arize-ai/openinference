from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    OpenAIServerModel,
)

from openinference.instrumentation.smolagents import SmolagentsInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

tracer = trace.get_tracer(__name__)


def run():
    model = OpenAIServerModel(model_id="gpt-5-nano")
    with tracer.start_span("main-operation"):
        manager_agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=model, max_steps=40)
        manager_agent.run(
            "How many seconds would it take for a leopard at full speed to run through "
            "Pont des Arts?"
        )


if __name__ == "__main__":
    run()
