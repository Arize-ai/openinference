from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterGrpc,
)
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import os

os.environ["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] = f"space_key=<space_key>,api_key=<api_key>"


def instrument():
    resource = Resource(
        attributes={
            "model_id": "offsite-llm-demo",
            "model_version": "v1",
        }
    )
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint="http://phoenix:6006/v1/traces")
    arize_exporter = OTLPSpanExporterGrpc(endpoint="https://otlp.arize.com/v1")
    span_processor = SimpleSpanProcessor(span_exporter=span_exporter)
    arize_span_processor = SimpleSpanProcessor(span_exporter=arize_exporter)
    tracer_provider.add_span_processor(span_processor=span_processor)
    tracer_provider.add_span_processor(span_processor=arize_span_processor)
    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    LlamaIndexInstrumentor().instrument()
