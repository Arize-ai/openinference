import asyncio
import json
import sys
from pathlib import Path

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.livekit import LiveKitInstrumentor

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from _stubs import install_livekit_stubs  # noqa: E402
from in_memory import _openinference_attrs, _run_calls  # noqa: E402


def main() -> None:
    install_livekit_stubs()
    in_memory_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))

    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:6006/v1/traces")
    batch_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(batch_processor)

    LiveKitInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    asyncio.run(_run_calls())

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
