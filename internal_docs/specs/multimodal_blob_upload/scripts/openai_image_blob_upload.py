# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai",
#     "opentelemetry-sdk",
#     "opentelemetry-exporter-otlp-proto-http",
#     "openinference-instrumentation",
#     "openinference-instrumentation-openai",
#     "openinference-semantic-conventions",
# ]
#
# [tool.uv.sources]
# openinference-instrumentation = { path = "../../../../python/openinference-instrumentation", editable = true }
# openinference-instrumentation-openai = { path = "../../../../python/instrumentation/openinference-instrumentation-openai", editable = true }
# openinference-semantic-conventions = { path = "../../../../python/openinference-semantic-conventions", editable = true }
# ///
"""How a custom blob-upload hook works.

A real vision call is made with an inline base64 PNG. Instead of recording
the base64 in the span, the instrumentation hands the decoded bytes to the
custom uploader below and records the URI it returns. Spans are printed and
exported to Phoenix (``PHOENIX_COLLECTOR_ENDPOINT``, default
http://localhost:6006).

Prerequisites: ``OPENAI_API_KEY``; optionally a local ``phoenix serve``.
Run:  uv run --script internal_docs/specs/multimodal_blob_upload/scripts/openai_image_blob_upload.py
"""

import base64
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path
from typing import Optional

import openai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import Blob, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor


class MyBlobUploader:
    """A custom blob-upload hook.

    ``BlobUploader`` is a Protocol — any object with these two methods
    works; there is no base class to inherit.

    - ``upload()`` must return the destination URI *immediately* and may do
      the actual write in the background — never block the instrumented
      call. Return None to reject the blob; the attribute is then redacted.
    - The ``Blob`` carries everything a store needs: decoded ``data`` bytes,
      ``mime_type``, ``modality``, ``content_sha256``, and ``attribute_key``
      (which span attribute the content came from).

    This one writes to a local directory — swap the body for S3/GCS or your
    artifact store. (A batteries-included fsspec version, ``FsspecBlobUploader``,
    ships in ``openinference.instrumentation``.)
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def upload(self, blob: Blob) -> Optional[str]:
        destination = self.directory / f"{blob.content_sha256[:16]}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob.data)
        print(
            f"uploaded {blob.mime_type} ({len(blob.data)} bytes) from {blob.attribute_key}"
        )
        return destination.as_uri()

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass  # flush pending writes here if uploading in the background


def make_png(size: int = 256) -> bytes:
    """A red circle on white — a real PNG the model can describe, no pillow needed."""
    rows = bytearray()
    for y in range(size):
        rows += b"\x00"  # PNG row filter: None
        for x in range(size):
            inside = (x - size / 2) ** 2 + (y - size / 2) ** 2 <= (size * 0.35) ** 2
            rows += b"\xdd\x22\x22" if inside else b"\xff\xff\xff"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set — this demo makes a real API call.")

    uploader = MyBlobUploader(Path(tempfile.mkdtemp(prefix="oi-blob-demo-")))
    config = TraceConfig(
        blob_uploader=uploader,
        base64_image_max_length=1_000,  # images over this externalize (default 32,000)
        enable_genai_semconv=True,  # dual-write: the image becomes a gen_ai 'uri' part
    )

    phoenix = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(f"{phoenix}/v1/traces"))
    )
    OpenAIInstrumentor().instrument(tracer_provider=provider, config=config)

    data_uri = "data:image/png;base64," + base64.b64encode(make_png()).decode()
    response = openai.OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What shape and color is this? One sentence.",
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )
    print(f"model answer: {response.choices[0].message.content}\n")

    uploader.shutdown()
    # Look for …message_content.image.image.url = file:///… (a URI, not base64)
    # and the {"type": "uri", "modality": "image", …} part in gen_ai.input.messages.
    for key, value in sorted((memory.get_finished_spans()[0].attributes or {}).items()):
        text = str(value)
        print(f"{key} = {text[:80] + '…' if len(text) > 80 else text}")
    print(
        f"\nimage bytes are under {uploader.directory}; spans are in Phoenix at {phoenix}"
    )


if __name__ == "__main__":
    main()
