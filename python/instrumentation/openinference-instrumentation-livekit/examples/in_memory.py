"""Replay LiveKit SDK calls and print the OpenInference attributes.

The script installs stand-ins for `Room`, `LocalParticipant`, and `TwirpClient`.
It does not open a socket and does not require the `livekit` wheel.
"""

import asyncio
import json
import sys
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation.livekit import LiveKitInstrumentor

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from _stubs import install_livekit_stubs  # noqa: E402


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
    return {k: v for k, v in attributes.items() if k in keep or k.startswith("openinference.")}


async def _run_calls() -> None:
    from livekit.api.twirp_client import TwirpClient
    from livekit.rtc.participant import LocalParticipant
    from livekit.rtc.room import Room

    room = Room()
    await room.connect("wss://demo.livekit.cloud", "secret-token-not-logged")
    await room.disconnect()

    participant = LocalParticipant()
    await participant.perform_rpc(
        destination_identity="agent",
        method="ping",
        payload='{"msg":"hi"}',
    )

    client = TwirpClient(host="https://demo.livekit.cloud", pkg="livekit")
    await client.request(
        "RoomService",
        "ListRooms",
        data=object(),
        headers={"Authorization": "Bearer secret-header"},
        response_class=object,
    )


def main() -> None:
    install_livekit_stubs()
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    LiveKitInstrumentor().instrument(tracer_provider=tracer_provider, skip_dep_check=True)
    asyncio.run(_run_calls())

    for span in exporter.get_finished_spans():
        oi = _openinference_attrs(dict(span.attributes or {}))
        print(span.name)
        print(json.dumps(oi, indent=2, default=str))


if __name__ == "__main__":
    main()
