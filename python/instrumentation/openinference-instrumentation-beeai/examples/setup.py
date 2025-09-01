import logging

from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

from openinference.instrumentation.beeai import BeeAIInstrumentor

logging.basicConfig(level=logging.INFO)


def setup_observability(endpoint: str = "http://localhost:6006/v1/traces") -> None:
    """
    Sets up OpenTelemetry with OTLP HTTP exporter and instruments the beeai framework.
    """
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
    tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace_api.set_tracer_provider(tracer_provider)

    BeeAIInstrumentor().instrument()
