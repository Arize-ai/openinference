# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.60.0",
#     "openinference-instrumentation-openai>=0.1.52",
#     "openinference-instrumentation",
#     "openinference-semantic-conventions>=0.1.30",
#     "opentelemetry-sdk>=1.42.0",
#     "opentelemetry-exporter-otlp-proto-http>=1.42.0",
#     "pillow>=10.0.0",
# ]
#
# [tool.uv.sources]
# openinference-instrumentation = { path = "../../../../python/openinference-instrumentation" }
# ///
"""Image path: a real auto-instrumented OpenAI vision call, redaction vs blob upload.

Runs the same chat-completions vision request twice through the released
openinference-instrumentation-openai auto-instrumentor. The app code never
changes — only the ``TraceConfig`` handed to the instrumentor does:

  run 1 — default config     the >32 KB base64 image is replaced with
                             ``__REDACTED__`` (today's released behavior).
  run 2 — blob-upload config ``TraceConfig(blob_uploader=...)`` — the same
                             attribute key records the blob store URI (a
                             repo-relative file path from the demo store).

The blob-upload pieces (``Blob``, ``BlobUploader``, the ``TraceConfig`` field
and mask policy) come from the **live** ``openinference-instrumentation``
package in this repo, resolved via the ``[tool.uv.sources]`` path above — this
script exercises the shipped code, not a local copy of it.

Prerequisites: OPENAI_API_KEY; a local ``phoenix serve`` (http://localhost:6006).
Run:  uv run --script internal_docs/specs/blob_uploads/scripts/image_blob_demo.py
"""

from __future__ import annotations

import atexit
import base64
import functools
import os
import random
import sys
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Optional

from openai import OpenAI
from openinference.instrumentation import Blob, BlobUploader, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import ImageAttributes, MessageContentAttributes
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from PIL import Image, ImageDraw

PROJECT_NAME = "blob-upload-image-demo"

_EXT_BY_MIME = {"image/png": ".png", "image/jpeg": ".jpg"}


class LocalBlobStore(BlobUploader):
    """Mock ``BlobUploader``: content-addressed files served over local HTTP.

    OpenInference ships no uploader implementation — this small store
    satisfies the shipped ``BlobUploader`` protocol for the demo (subclassing
    the protocol is optional but makes the contract explicit). It writes
    content-addressed files under ``scripts/blob_store/`` and returns an
    ``http://localhost:<port>/…`` URL served by a background thread, so the
    recorded reference is a real, renderable URL — Phoenix can display the
    image as long as this script keeps serving. Writes synchronously — fine
    for a demo; real implementations move bytes on a background worker.
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        self.root_dir = root_dir or Path(__file__).parent / "blob_store"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        handler = functools.partial(SimpleHTTPRequestHandler, directory=str(self.root_dir))
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        self.base_url = f"http://localhost:{self._server.server_address[1]}"
        print(f"[blob] serving {self.root_dir} at {self.base_url}")

    def upload(self, blob: Blob) -> Optional[str]:
        name = blob.content_sha256[:20] + _EXT_BY_MIME.get(blob.mime_type, ".bin")
        path = self.root_dir / name
        if not path.exists():  # content-addressed dedup
            path.write_bytes(blob.data)
            print(f"[blob] stored {blob.modality} ({len(blob.data):,} B) → {path.name}")
        return f"{self.base_url}/{name}"

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        self._server.shutdown()


def make_demo_png() -> bytes:
    """A labeled banner over seeded RGB noise — noise is incompressible, so the
    PNG lands in the hundreds-of-KB range a real photo occupies, far over the
    32,000-char base64 budget TraceConfig allows an image today."""
    rng = random.Random(42)
    width, height = 640, 400
    img = Image.frombytes("RGB", (width, height), rng.randbytes(width * height * 3))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, width, 56], fill=(16, 24, 48))
    draw.text((16, 12), "OpenInference blob-upload demo", fill=(240, 240, 255))
    draw.text((16, 32), "synthetic test pattern (seeded noise)", fill=(160, 170, 200))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ask_about_image(data_uri: str) -> str:
    """The real app: one chat-completions vision call."""
    response = OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


IMAGE_URL_SUFFIX = (
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)


def print_spans(memory: InMemorySpanExporter, label: str, since: int) -> int:
    """Print each span from this run, attribute by attribute."""
    spans = memory.get_finished_spans()[since:]
    for span in spans:
        attributes = span.attributes or {}
        total = sum(len(k) + len(str(v)) for k, v in attributes.items())
        print(f"\n── {span.name} — {label}  ({len(attributes)} attrs, {total:,} B) ──")
        for key in sorted(attributes):
            text = str(attributes[key]).replace("\n", "\\n")
            if len(text) > 76:
                text = f"{text[:76]}… ({len(text):,} chars)"
            print(f"  {key} = {text}")
    return len(memory.get_finished_spans())


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set — this demo makes real vision calls.")

    png = make_demo_png()
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    print(
        f"generated image: {len(png):,} B PNG → {len(data_uri):,} chars as a data URI"
    )

    phoenix = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": PROJECT_NAME})
    )
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(f"{phoenix}/v1/traces"))
    )

    store = LocalBlobStore()
    # Implementations own their exit flush; the demo also keeps the HTTP
    # server alive below so Phoenix can render the URLs.
    atexit.register(store.shutdown)
    seen = 0
    for label, config in [
        ("default config (image __REDACTED__)", TraceConfig()),
        ("blob upload (external URI)", TraceConfig(blob_uploader=store)),
    ]:
        OpenAIInstrumentor().instrument(tracer_provider=provider, config=config)
        answer = ask_about_image(data_uri)
        OpenAIInstrumentor().uninstrument()
        print(f"\n[{label}] model: {answer}")
        provider.force_flush()
        seen = print_spans(memory, label, seen)

    provider.shutdown()

    print(f"\nPhoenix: {phoenix}  → project {PROJECT_NAME!r}")
    print("Compare the two runs' ChatCompletion spans: message_content.image.image.url")
    print(f"is __REDACTED__ in the first and a {store.base_url}/… URL in the second,")
    print("which Phoenix renders inline while this script keeps serving.")
    if sys.stdin.isatty():
        input("Serving the blob store — press Enter to stop and exit... ")


if __name__ == "__main__":
    main()
