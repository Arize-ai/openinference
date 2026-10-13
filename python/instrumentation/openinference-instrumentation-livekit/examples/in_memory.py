"""Connect to LiveKit and print the OpenInference attributes.

The LiveKit calls live in ``livekit_calls.py``. They use the real ``livekit``
and ``livekit-api`` packages. Set ``LIVEKIT_URL``, ``LIVEKIT_API_KEY``, and
``LIVEKIT_API_SECRET`` before running.
"""

import asyncio
import json

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.livekit import LiveKitInstrumentor


def _openinference_attrs(attributes: dict[str, object] | None) -> dict[str, object]:
    if not attributes:
        return {}
    from openinference.semconv.trace import SpanAttributes

    keep = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        SpanAttributes.INPUT_VALUE,
        SpanAttributes.INPUT_MIME_TYPE,
        SpanAttributes.OUTPUT_VALUE,
        SpanAttributes.OUTPUT_MIME_TYPE,
    }
    return {
        key: value
        for key, value in attributes.items()
        if key in keep or key.startswith("openinference.")
    }


def main() -> None:
    from livekit_calls import run_livekit_calls

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    LiveKitInstrumentor().instrument(tracer_provider=tracer_provider)
    asyncio.run(run_livekit_calls())

    for span in exporter.get_finished_spans():
        oi = _openinference_attrs(dict(span.attributes or {}))
        print(span.name)
        print(json.dumps(oi, indent=2, default=str))


if __name__ == "__main__":
    main()
