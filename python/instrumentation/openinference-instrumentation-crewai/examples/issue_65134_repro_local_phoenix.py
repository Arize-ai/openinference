"""
Repro for CrewAI task-span `input.value.agent`, exported to local Phoenix.

Run local Phoenix first:
    python -m phoenix.server.main serve

Then run:
    python \
      python/instrumentation/openinference-instrumentation-crewai/examples/issue_65134_repro_local_phoenix.py
"""

from __future__ import annotations

from _issue_65134_repro_common import (
    PROJECT_NAME,
    add_console_exporter,
    run_repro,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

PHOENIX_COLLECTOR_ENDPOINT = "http://127.0.0.1:6006/v1/traces"


def build_tracer_provider() -> trace_sdk.TracerProvider:
    tracer_provider = trace_sdk.TracerProvider(
        resource=Resource.create({"service.name": PROJECT_NAME})
    )
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_COLLECTOR_ENDPOINT))
    )
    add_console_exporter(tracer_provider)
    return tracer_provider


def main() -> None:
    print(f"Sending traces to local Phoenix at {PHOENIX_COLLECTOR_ENDPOINT}")
    print("View traces at http://127.0.0.1:6006\n")
    run_repro(tracer_provider=build_tracer_provider())


if __name__ == "__main__":
    main()
