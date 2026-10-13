"""Connect to LiveKit, print the spans, and export them to a local Phoenix.

Start Phoenix on port 6006 first if you want the export to land.
The LiveKit calls are the same as ``in_memory.py``.
"""

import asyncio
import json

from in_memory import _openinference_attrs
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.livekit import LiveKitInstrumentor


def main() -> None:
    from livekit_calls import run_livekit_calls

    in_memory_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))

    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
    batch_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(batch_processor)

    LiveKitInstrumentor().instrument(tracer_provider=tracer_provider)
    asyncio.run(run_livekit_calls())

    try:
        batch_processor.force_flush(timeout_millis=3000)
    except Exception as exc:
        print(f"OTLP export failed (is Phoenix running on :6006?): {exc}")

    for span in in_memory_exporter.get_finished_spans():
        oi = _openinference_attrs(dict(span.attributes or {}))
        print(span.name)
        print(json.dumps(oi, indent=2, default=str))


if __name__ == "__main__":
    main()
