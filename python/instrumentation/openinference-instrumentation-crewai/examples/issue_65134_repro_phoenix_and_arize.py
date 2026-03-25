"""
Repro for CrewAI task-span `input.value.agent`, exported to Arize and local Phoenix.

Run local Phoenix first:
    python -m phoenix.server.main serve

Then fill in the Arize placeholders below and run:
    python \
      python/instrumentation/openinference-instrumentation-crewai/examples/issue_65134_repro_phoenix_and_arize.py
"""

from __future__ import annotations

from _issue_65134_repro_common import (
    PROJECT_NAME,
    add_console_exporter,
    require_value,
    run_repro,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

PHOENIX_COLLECTOR_ENDPOINT = "http://127.0.0.1:6006/v1/traces"
ARIZE_SPACE_ID = "INSERT_ARIZE_SPACE_ID_HERE"
ARIZE_API_KEY = "INSERT_ARIZE_API_KEY_HERE"


def build_tracer_provider():
    try:
        from arize.otel import register as register_arize
    except ImportError as exc:
        raise RuntimeError("Install `arize-otel` to run the Arize example.") from exc

    tracer_provider = register_arize(
        space_id=require_value("ARIZE_SPACE_ID", ARIZE_SPACE_ID),
        api_key=require_value("ARIZE_API_KEY", ARIZE_API_KEY),
        project_name=PROJECT_NAME,
    )
    tracer_provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=PHOENIX_COLLECTOR_ENDPOINT))
    )
    add_console_exporter(tracer_provider)
    return tracer_provider


def main() -> None:
    print(f"Sending traces to Arize project `{PROJECT_NAME}` and {PHOENIX_COLLECTOR_ENDPOINT}")
    print("View local traces at http://127.0.0.1:6006\n")
    run_repro(tracer_provider=build_tracer_provider())


if __name__ == "__main__":
    main()
