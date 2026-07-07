# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai",
#     "opentelemetry-api",
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
"""OpenAI image blob-upload demo against the live API, exported to Phoenix.

Sends a real vision chat completion whose image arrives as a base64 data
URI, and proves the experimental blob-upload design end-to-end:

  1. The instrumentation detects the oversized base64 image at capture
     time, uploads the decoded bytes to storage, and records only the
     destination URI in ``…message_content.image.image.url``.
  2. Scenario A uses the built-in ``FsspecBlobUploader`` (content-addressed
     ``{sha256}.png`` destinations, background bounded queue). Scenario B
     implements a custom ``BlobUploader`` hook from scratch.
  3. Spans are captured in-memory for PASS/FAIL assertions, printed, and
     exported to a local Phoenix so the attributes can be inspected in the
     UI — including the OTel GenAI dual-write, where the externalized
     image appears as a spec-conformant ``uri`` message part.

The assertions make this a real proof: the PNG bytes exist only in this
process, so if capture, threshold detection, upload, or URI substitution
failed anywhere, the span would carry base64 or ``__REDACTED__`` and the
sha256 comparison against the uploaded file would fail.

Prerequisites: ``OPENAI_API_KEY`` set; Phoenix running locally
(``phoenix serve``) or ``PHOENIX_COLLECTOR_ENDPOINT`` pointing elsewhere.

Run:  uv run --script internal_docs/specs/multimodal_blob_upload/scripts/openai_image_blob_upload.py
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import struct
import sys
import tempfile
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import openai
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import Blob, FsspecBlobUploader, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from openinference.semconv.trace import (
    ImageAttributes,
    MessageAttributes,
    MessageContentAttributes,
    SpanAttributes,
)

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
PROJECT_NAME = "blob-upload-image-demo"
IMAGE_URL_SUFFIX = (
    f"{MessageContentAttributes.MESSAGE_CONTENT_IMAGE}.{ImageAttributes.IMAGE_URL}"
)

# ---------------------------------------------------------------------------
# Demo image: a red circle on white, generated as a real PNG (no pillow)
# ---------------------------------------------------------------------------


def make_png(width: int = 256, height: int = 256) -> bytes:
    radius_squared = (min(width, height) * 0.35) ** 2
    rows = bytearray()
    for y in range(height):
        rows += b"\x00"  # PNG filter type: None
        for x in range(width):
            dx, dy = x - width / 2, y - height / 2
            inside = dx * dx + dy * dy <= radius_squared
            rows += b"\xdd\x22\x22" if inside else b"\xff\xff\xff"

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(rows)))
        + chunk(b"IEND", b"")
    )


PNG_BYTES = make_png()
PNG_SHA256 = hashlib.sha256(PNG_BYTES).hexdigest()
PNG_DATA_URI = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


# ---------------------------------------------------------------------------
# Scenario B subject: a custom BlobUploader hook
# ---------------------------------------------------------------------------
#
# ``BlobUploader`` is a runtime-checkable Protocol — no base class. The
# contract: ``upload(blob)`` returns the destination URI *immediately* and
# performs the actual write out-of-band (or fast enough not to matter);
# return None to reject, and the instrumentation falls back to redaction.
# ``shutdown(timeout_sec)`` flushes pending writes.


@dataclass
class ManifestBlobUploader:
    """Writes blobs under a directory and keeps a manifest.json sidecar.

    Swap the body of ``upload`` for boto3/S3, GCS, or an artifact store —
    the URI you return is what lands in the span attribute.
    """

    directory: Path
    manifest: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)

    def upload(self, blob: Blob) -> Optional[str]:
        extension = {"image/png": ".png", "image/jpeg": ".jpg"}.get(
            blob.mime_type, ".bin"
        )
        destination = self.directory / f"{blob.content_sha256[:16]}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob.data)
        self.manifest[destination.name] = {
            "mime_type": blob.mime_type,
            "modality": blob.modality,
            "sha256": blob.content_sha256,
            "captured_from": blob.attribute_key,
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2)
        )
        return destination.as_uri()

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass  # nothing queued; a real remote uploader would flush here


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def run_scenario(config: TraceConfig, phoenix_exporter: OTLPSpanExporter) -> Any:
    """Instrument, make one real vision call, export to Phoenix, return the span."""
    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": PROJECT_NAME})
    )
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=provider, config=config)
    try:
        response = openai.OpenAI().chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "What shape is in this image, and what color is it? "
                            "Answer in one short sentence.",
                        },
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URI}},
                    ],
                }
            ],
        )
    finally:
        instrumentor.uninstrument()
    print(f"model answer: {response.choices[0].message.content}")
    spans = memory_exporter.get_finished_spans()
    phoenix_exporter.export(spans)  # same spans, now visible in the Phoenix UI
    return spans[0]


def show_span(span: Any) -> None:
    print(f"\nspan: {span.name}")
    for key in sorted(span.attributes or {}):
        value = str((span.attributes or {})[key])
        if len(value) > 96:
            value = f"{value[:80]}… ({len(value)} chars)"
        print(f"  {key} = {value}")


def check(description: str, passed: bool, failures: list) -> None:
    print(f"  {'PASS' if passed else 'FAIL'}  {description}")
    if not passed:
        failures.append(description)


def assert_externalized(span: Any, upload_dir: Path, failures: list) -> None:
    attributes = dict(span.attributes or {})
    image_url_key = (
        f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.{MessageAttributes.MESSAGE_CONTENTS}.1."
        f"{IMAGE_URL_SUFFIX}"
    )
    url = str(attributes.get(image_url_key, ""))
    check(
        "image attribute holds a URI, not base64",
        bool(url) and not url.startswith("data:"),
        failures,
    )
    check("image attribute was not redacted", url != "__REDACTED__", failures)

    local_path = Path(urlparse(url).path) if url.startswith("file:") else Path(url)
    uploaded = (
        local_path.exists()
        and hashlib.sha256(local_path.read_bytes()).hexdigest() == PNG_SHA256
    )
    check(
        "uploaded file exists and sha256 matches the original PNG", uploaded, failures
    )
    check(
        "uploaded files live under the scenario's upload dir",
        str(local_path).startswith(str(upload_dir)),
        failures,
    )

    if genai_messages := attributes.get("gen_ai.input.messages"):
        parts = json.loads(str(genai_messages))[0]["parts"]
        image_parts = [p for p in parts if p.get("modality") == "image"]
        check(
            "gen_ai.input.messages carries the image as a 'uri' part",
            bool(image_parts) and all(p["type"] == "uri" for p in image_parts),
            failures,
        )


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "OPENAI_API_KEY is not set — this demo makes real API calls.",
            file=sys.stderr,
        )
        return 2

    phoenix_base_url = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    phoenix_exporter = OTLPSpanExporter(endpoint=f"{phoenix_base_url}/v1/traces")

    workdir = Path(
        os.environ.get("OPENINFERENCE_BLOB_UPLOAD_BASE_PATH")
        or tempfile.mkdtemp(prefix="oi-image-blob-upload-")
    )
    print(f"image as a data URI is {len(PNG_DATA_URI)} chars", end="; ")
    print("the config below externalizes images over 1,000 chars")
    failures: list = []

    # ---- Scenario A: built-in local upload --------------------------------
    # Zero-code equivalent: OPENINFERENCE_BLOB_UPLOAD_BASE_PATH=/var/oi-media (or s3://…)
    print(
        f"\n{'=' * 78}\nScenario A — FsspecBlobUploader (local directory)\n{'=' * 78}"
    )
    fsspec_dir = workdir / "fsspec-upload"
    uploader_a = FsspecBlobUploader(base_path=str(fsspec_dir))
    span = run_scenario(
        TraceConfig(
            blob_uploader=uploader_a,
            base64_image_max_length=1_000,  # demo threshold; default is 32,000
            enable_genai_semconv=True,  # dual-write: externalized image => 'uri' part
        ),
        phoenix_exporter,
    )
    uploader_a.shutdown()  # flush the background queue before checking files
    show_span(span)
    print()
    assert_externalized(span, fsspec_dir, failures)

    # ---- Scenario B: custom uploader hook ----------------------------------
    print(f"\n{'=' * 78}\nScenario B — custom ManifestBlobUploader\n{'=' * 78}")
    custom_dir = workdir / "custom-upload"
    uploader_b = ManifestBlobUploader(directory=custom_dir)
    span = run_scenario(
        TraceConfig(blob_uploader=uploader_b, base64_image_max_length=1_000),
        phoenix_exporter,
    )
    uploader_b.shutdown()
    show_span(span)
    print()
    assert_externalized(span, custom_dir, failures)
    print("\nmanifest.json:")
    print((custom_dir / "manifest.json").read_text())

    print(f"uploaded artifacts kept under {workdir} — the .png files are viewable")
    print(
        f"open {phoenix_base_url} and select project '{PROJECT_NAME}' to inspect the spans"
    )
    if failures:
        print(f"\n{len(failures)} assertion(s) FAILED", file=sys.stderr)
        return 1
    print("\nall assertions PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
