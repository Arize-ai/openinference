import os

from langchain_openai import ChatOpenAI
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
trace_api.set_tracer_provider(tracer_provider=tracer_provider)

# Logs to the Phoenix Collector if running locally
if os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"):
    endpoint = os.environ["PHOENIX_COLLECTOR_ENDPOINT"] + "/v1/traces"
    span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))
else:
    span_console_exporter = ConsoleSpanExporter()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_console_exporter))

LangChainInstrumentor().instrument()


if __name__ == "__main__":
    for chunk in ChatOpenAI(model_name="gpt-3.5-turbo").stream("Write a haiku."):
        print(chunk.content, end="", flush=True)
