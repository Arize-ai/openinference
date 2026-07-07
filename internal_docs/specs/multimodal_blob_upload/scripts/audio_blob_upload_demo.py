# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai",
#     "httpx",
#     "openinference-instrumentation",
#     "openinference-instrumentation-openai",
#     "openinference-semantic-conventions",
#     "opentelemetry-sdk",
# ]
#
# [tool.uv.sources]
# openinference-instrumentation = { path = "../../../../python/openinference-instrumentation", editable = true }
# openinference-instrumentation-openai = { path = "../../../../python/instrumentation/openinference-instrumentation-openai", editable = true }
# openinference-semantic-conventions = { path = "../../../../python/openinference-semantic-conventions", editable = true }
# ///
"""
End-to-end walkthrough: capturing large audio content with blob upload.

This demo sends a chat completion containing base64 WAV audio (the
``input_audio`` content part of ``gpt-4o-audio-preview``-style models) and
receives an audio response. Instead of recording multi-megabyte base64 data
URIs in span attributes, the instrumentation uploads the decoded bytes to
storage and records only the destination URI.

It demonstrates, in order:

1. Uploading to a local directory with the built-in ``FsspecBlobUploader``.
2. Implementing a custom ``BlobUploader`` hook (a manifest-keeping local
   uploader — the same shape you would use for S3 presigned URLs, an
   artifact store, etc.).
3. Inspecting the resulting spans with an ``InMemorySpanExporter`` —
   including the OTel GenAI dual-write (``gen_ai.input.messages``), where
   the externalized audio appears as a spec-conformant ``uri`` part.

The OpenAI HTTP layer is mocked with ``httpx.MockTransport``, so the script
runs offline with no API key, and the local blob destinations need no
fsspec install. The uv sources above point at this branch's editable
packages, so the script always exercises the in-repo implementation.

Run:  uv run --script internal_docs/specs/multimodal_blob_upload/scripts/audio_blob_upload_demo.py
"""

import base64
import io
import json
import math
import struct
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import httpx
import openai
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from openinference.instrumentation import Blob, FsspecBlobUploader, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor

# ---------------------------------------------------------------------------
# Demo audio: generate small, real (playable) WAV clips
# ---------------------------------------------------------------------------


def make_wav(frequency_hz: float, seconds: float = 0.5, rate: int = 8_000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        for i in range(int(seconds * rate)):
            sample = int(0.5 * 32767 * math.sin(2 * math.pi * frequency_hz * i / rate))
            wav.writeframes(struct.pack("<h", sample))
    return buffer.getvalue()


INPUT_WAV_B64 = base64.b64encode(make_wav(440.0)).decode()  # the "question" audio
OUTPUT_WAV_B64 = base64.b64encode(make_wav(660.0)).decode()  # the "answer" audio


# ---------------------------------------------------------------------------
# Offline OpenAI client: MockTransport returns a canned audio response
# ---------------------------------------------------------------------------


def mock_openai_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-demo",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "gpt-4o-audio-preview",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "audio": {
                            "id": "audio_demo123",
                            "data": OUTPUT_WAV_B64,
                            "transcript": "That tone was an A above middle C (440 Hz).",
                            "expires_at": 1_700_003_600,
                        },
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 30,
                "total_tokens": 72,
                "prompt_tokens_details": {"audio_tokens": 25},
                "completion_tokens_details": {"audio_tokens": 20},
            },
        },
    )


def make_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key="sk-demo-offline",
        http_client=httpx.Client(transport=httpx.MockTransport(mock_openai_response)),
    )


# ---------------------------------------------------------------------------
# Part 2 subject: a custom BlobUploader hook
# ---------------------------------------------------------------------------
#
# ``BlobUploader`` is a runtime-checkable Protocol — no base class to
# inherit. The contract:
#
#   upload(blob) -> Optional[str]
#       Return the destination URI *immediately* and perform the actual
#       write out-of-band (or fast enough not to matter, as here). Return
#       None to reject the blob — the instrumentation then falls back to
#       the standard redaction (``__REDACTED__``). Never block the
#       instrumented call path on remote storage.
#
#   shutdown(timeout_sec) -> None
#       Flush pending writes. Called by your application at exit.
#
# ``blob`` carries: decoded ``data`` bytes, ``mime_type``, ``modality``
# ("image"|"audio"|"video"|"document"), ``content_sha256`` (precomputed),
# and ``attribute_key`` (which span attribute the content came from).


@dataclass
class ManifestBlobUploader:
    """Writes blobs under a directory and keeps a manifest.json sidecar.

    Swap the body of ``upload`` for boto3/S3, GCS, or an artifact store —
    the URI you return is what lands in the span attribute.
    """

    directory: Path
    manifest: Dict[str, Dict[str, Optional[str]]] = field(default_factory=dict)

    def upload(self, blob: Blob) -> Optional[str]:
        extension = {"audio/wav": ".wav", "application/pdf": ".pdf"}.get(
            blob.mime_type, ".bin"
        )
        destination = self.directory / f"{blob.content_sha256[:16]}{extension}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob.data)  # local write: cheap enough to do inline
        self.manifest[destination.name] = {
            "mime_type": blob.mime_type,
            "modality": blob.modality,
            "sha256": blob.content_sha256,
            "captured_from": blob.attribute_key,
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2)
        )
        return destination.as_uri()  # file:///…/<sha>.wav

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass  # nothing queued; a real remote uploader would flush here


# ---------------------------------------------------------------------------
# Span inspection helpers (Part 3)
# ---------------------------------------------------------------------------


def show_spans(exporter: InMemorySpanExporter, title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    for span in exporter.get_finished_spans():
        print(f"span: {span.name}")
        for key in sorted(span.attributes or {}):
            value = str((span.attributes or {})[key])
            if len(value) > 96:  # keep base64/JSON readable
                value = f"{value[:80]}… ({len(value)} chars)"
            print(f"  {key} = {value}")


def show_genai_messages(exporter: InMemorySpanExporter) -> None:
    attributes = exporter.get_finished_spans()[0].attributes or {}
    for key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        if key in attributes:
            print(f"\n{key} (dual-write, externalized audio is a 'uri' part):")
            print(json.dumps(json.loads(str(attributes[key])), indent=2))


def show_uploaded_files(directory: Path) -> None:
    print(f"\nfiles under {directory}:")
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            print(f"  {path.name}  ({path.stat().st_size} bytes)")


def run_instrumented_call(config: TraceConfig) -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = OpenAIInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, config=config)
    try:
        make_client().chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What note is this?"},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": INPUT_WAV_B64, "format": "wav"},
                        },
                    ],
                }
            ],
        )
    finally:
        instrumentor.uninstrument()
    return exporter


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="oi-blob-upload-demo-"))
    data_uri_length = len(f"data:audio/wav;base64,{INPUT_WAV_B64}")
    print(f"demo audio as a data URI is {data_uri_length} chars", end="; ")
    print("the config below externalizes anything over 1,000 chars")

    # ---- Part 1: built-in local upload -----------------------------------
    # FsspecBlobUploader handles s3://, gs://, file://, and plain local
    # paths (local needs no fsspec install). Destinations are
    # content-addressed ({sha256}.wav) and written by a bounded background
    # queue — identical audio sent twice uploads once.
    #
    # Zero-code equivalent:
    #   export OPENINFERENCE_BLOB_UPLOAD_BASE_PATH=/var/oi-media  (or s3://…)
    fsspec_dir = workdir / "fsspec-upload"
    fsspec_uploader = FsspecBlobUploader(base_path=str(fsspec_dir))
    exporter = run_instrumented_call(
        TraceConfig(
            blob_uploader=fsspec_uploader,
            base64_media_max_length=1_000,  # demo threshold; default is 32,000
            enable_genai_semconv=True,
        )
    )
    fsspec_uploader.shutdown()  # flush the background queue before reading files
    show_spans(exporter, "Part 1 — FsspecBlobUploader (local directory)")
    show_genai_messages(exporter)
    show_uploaded_files(fsspec_dir)

    # ---- Part 2: custom uploader hook -------------------------------------
    custom_dir = workdir / "custom-upload"
    custom_uploader = ManifestBlobUploader(directory=custom_dir)
    exporter = run_instrumented_call(
        TraceConfig(
            blob_uploader=custom_uploader,
            base64_media_max_length=1_000,
        )
    )
    custom_uploader.shutdown()
    show_spans(exporter, "Part 2 — custom ManifestBlobUploader")
    show_uploaded_files(custom_dir)
    print("\nmanifest.json:")
    print((custom_dir / "manifest.json").read_text())

    # ---- Part 3 ran throughout: spans came from InMemorySpanExporter ------
    print(f"\nAll artifacts kept under {workdir} — the .wav files are playable.")


if __name__ == "__main__":
    main()
