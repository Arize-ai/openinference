"""Shared helpers for the blob-upload before/after demo scripts.

Each demo script:

  1. Builds "before" spans that inline large multimodal content as
     ``data:<mime>;base64,...`` attribute values — byte-for-byte the shapes
     produced by released OpenInference code today (openai-agents realtime
     audio from PR #3173, image content via ``OITracer`` + ``TraceConfig``).
  2. Builds "after" spans where the decoded bytes are handed to the proposed
     ``BlobUploader`` at capture time and the span attribute records only the
     destination URI.
  3. Exports both to a local Phoenix so the difference is visible side by side
     in one project.
  4. Prints PASS/FAIL assertions proving the after-path preserves content
     (byte-equal fetch through the blob store) at a fraction of the span size.

The ``Blob`` / ``BlobUploader`` definitions below are the *proposed* interface
from ../blob_uploads.md. ``LocalHttpBlobUploader`` is a demo backend only —
local filesystem storage plus a tiny static HTTP server so the Phoenix UI
(a browser) can actually resolve the URIs. Production backends (S3, GCS, ...)
are explicitly out of scope.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import math
import os
import queue
import re
import struct
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# ---------------------------------------------------------------------------
# Proposed interface (mirrors ../blob_uploads.md §4 verbatim)
#
# These classes are what the techspec proposes adding to
# ``openinference.instrumentation`` (as ``openinference.instrumentation.blobs``).
# The demo defines them locally because the released packages pinned by these
# scripts do not ship them yet.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Blob:
    """A finalized binary payload captured by an instrumentor."""

    data: bytes
    """Decoded bytes — never base64 text."""

    mime_type: str
    """IANA media type, e.g. ``audio/wav``, ``image/png``."""

    attribute_key: Optional[str] = None
    """Span attribute the reference will be written to, e.g.
    ``input.audio.url`` — lets uploaders partition/label storage."""

    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    """Hex-encoded ids of the owning span, when available at the call site."""


@runtime_checkable
class BlobUploader(Protocol):
    """Pluggable destination for large multimodal payloads.

    ``upload`` MUST return quickly: implementations compute the destination
    URI synchronously and transfer bytes on a background worker. A ``None``
    return means the uploader cannot accept the blob (backpressure, shutdown)
    and the caller MUST fall back to today's inline behavior.
    """

    def upload(self, blob: Blob) -> Optional[str]: ...

    def force_flush(self, timeout_s: float = 10.0) -> bool: ...

    def shutdown(self) -> None: ...


# ---------------------------------------------------------------------------
# Demo backend: local filesystem + static HTTP server
# ---------------------------------------------------------------------------

_EXT_BY_MIME = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "application/pdf": ".pdf",
}

DEFAULT_BLOB_PORT = int(os.environ.get("BLOB_DEMO_PORT", "8321"))
BLOB_STORE_DIR = Path(__file__).parent / "blob_store"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


class LocalHttpBlobUploader:
    """Filesystem-backed demo uploader with async writes + HTTP read path.

    - Destination URI is computed synchronously from a content hash
      (``sha256[:20]`` + extension) so ``upload()`` never blocks on I/O and
      identical payloads dedup to one object.
    - Bytes are written by a daemon worker thread fed by a bounded queue;
      ``upload()`` returns ``None`` on queue overflow (caller falls back to
      the inline path) instead of blocking the app's hot path.
    - Files are served at ``http://127.0.0.1:<port>/<name>`` so the Phoenix
      UI can render the URIs. If the port is already bound, a previous demo
      run is still serving the same directory and we reuse it.
    """

    def __init__(
        self,
        root_dir: Path = BLOB_STORE_DIR,
        port: int = DEFAULT_BLOB_PORT,
        queue_capacity: int = 64,
    ) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = f"http://127.0.0.1:{port}"
        self._queue: "queue.Queue[tuple[Path, bytes]]" = queue.Queue(
            maxsize=queue_capacity
        )
        self._dropped = 0
        self._shutdown = False
        self._worker = threading.Thread(
            target=self._drain, daemon=True, name="blob-uploader"
        )
        self._worker.start()
        self._server: Optional[ThreadingHTTPServer] = None
        try:
            self._server = ThreadingHTTPServer(
                ("127.0.0.1", port),
                partial(_QuietHandler, directory=str(self.root_dir)),
            )
            threading.Thread(
                target=self._server.serve_forever, daemon=True, name="blob-http"
            ).start()
            print(f"[blob] serving {self.root_dir} at {self.base_url}")
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            # A previous demo run is still serving this directory; new files
            # land in the same dir, so its server picks them up on request.
            print(f"[blob] reusing blob server already listening at {self.base_url}")

    # -- BlobUploader protocol ------------------------------------------------

    def upload(self, blob: Blob) -> Optional[str]:
        if self._shutdown:
            return None
        ext = _EXT_BY_MIME.get(blob.mime_type, ".bin")
        name = hashlib.sha256(blob.data).hexdigest()[:20] + ext
        path = self.root_dir / name
        uri = f"{self.base_url}/{name}"
        if path.exists():  # content-addressed dedup
            return uri
        try:
            self._queue.put_nowait((path, blob.data))
        except queue.Full:
            self._dropped += 1
            return None
        return uri

    def force_flush(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while self._queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def shutdown(self) -> None:
        self.force_flush()
        self._shutdown = True

    def stop_serving(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None

    # -- internals ------------------------------------------------------------

    def _drain(self) -> None:
        while True:
            path, data = self._queue.get()
            try:
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(data)
                tmp.replace(path)  # atomic: readers never see partial blobs
            except OSError as exc:
                print(
                    f"[blob] WARN failed to persist {path.name}: {exc}", file=sys.stderr
                )
            finally:
                self._queue.task_done()


# ---------------------------------------------------------------------------
# Data-URI helpers
#
# ``pcm16_to_wav_data_uri`` and ``truncate_audio_data_uri`` are copied from
# openinference-instrumentation-openai-agents ``_realtime.py`` (PR #3173) so
# the "before" spans are byte-for-byte what that instrumentor emits today.
# ---------------------------------------------------------------------------

# OpenAI Realtime API streams 24 kHz mono PCM16 in both directions.
SAMPLE_RATE_HZ = 24_000
SAMPLE_WIDTH_BYTES = 2
NUM_CHANNELS = 1

DEFAULT_BASE64_AUDIO_MAX_LENGTH = (
    32_000  # OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH default
)
DEFAULT_BASE64_IMAGE_MAX_LENGTH = (
    32_000  # OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH default
)

_DATA_URI_PATTERN = re.compile(
    r"^data:(?P<mime>[^;,]+);base64,(?P<payload>.*)$", re.DOTALL
)


def pcm16_to_wav_data_uri(
    pcm_bytes: bytes,
    sample_rate: int = SAMPLE_RATE_HZ,
    sample_width: int = SAMPLE_WIDTH_BYTES,
    channels: int = NUM_CHANNELS,
) -> str:
    """Encode raw PCM16 bytes to a WAV data: URI (audio/wav)."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm_bytes)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:audio/wav;base64,{b64}"


def pcm16_to_wav_bytes(pcm_bytes: bytes) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(NUM_CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav.setframerate(SAMPLE_RATE_HZ)
        wav.writeframes(pcm_bytes)
    return buf.getvalue()


def truncate_audio_data_uri(uri: str, max_length: int) -> str:
    """Truncate the base64 payload of a data: URI, preserving the prefix.

    Mirrors PR #3173: the result is intentionally NOT valid base64 — the
    payload is cut mid-stream, which is exactly why inline capture cannot
    preserve production audio.
    """
    prefix, sep, payload = uri.partition(";base64,")
    if not sep:
        return uri[:max_length]
    return prefix + sep + payload[:max_length]


def decode_data_uri(uri: str) -> tuple[str, bytes]:
    """Split a ``data:<mime>;base64,<payload>`` URI into (mime, bytes)."""
    match = _DATA_URI_PATTERN.match(uri)
    if not match:
        raise ValueError("not a base64 data URI")
    return match.group("mime"), base64.b64decode(match.group("payload"))


def synth_pcm16(
    segments: list[tuple[float, float]],
    sample_rate: int = SAMPLE_RATE_HZ,
    amplitude: float = 0.4,
) -> bytes:
    """Deterministic PCM16 mono synth: ``segments`` is (frequency_hz, seconds).

    Produces a little melody so the recovered blob is audibly a real WAV —
    a stand-in for the microphone / assistant audio a realtime session buffers.
    """
    frames = bytearray()
    for freq, seconds in segments:
        n = int(seconds * sample_rate)
        for i in range(n):
            t = i / sample_rate
            fade = min(1.0, (n - i) / (0.01 * sample_rate), i / (0.01 * sample_rate))
            sample = amplitude * fade * math.sin(2 * math.pi * freq * t)
            frames += struct.pack("<h", int(sample * 32767))
    return bytes(frames)


# ---------------------------------------------------------------------------
# OTel + Phoenix setup
# ---------------------------------------------------------------------------


@dataclass
class TracingCtx:
    provider: TracerProvider
    memory_exporter: InMemorySpanExporter
    project_name: str
    phoenix_base_url: str

    def finished_spans(self) -> tuple[ReadableSpan, ...]:
        return self.memory_exporter.get_finished_spans()

    def span_named(self, name: str) -> ReadableSpan:
        for span in self.finished_spans():
            if span.name == name:
                return span
        raise RuntimeError(f"span {name!r} was not captured")

    def shutdown(self) -> None:
        self.provider.force_flush()
        self.provider.shutdown()


def setup_tracing(project_name: str) -> TracingCtx:
    base_url = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    resource = Resource.create({"openinference.project.name": project_name})
    provider = TracerProvider(resource=resource)
    memory_exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory_exporter))
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=f"{base_url}/v1/traces"))
    )
    return TracingCtx(
        provider=provider,
        memory_exporter=memory_exporter,
        project_name=project_name,
        phoenix_base_url=base_url,
    )


def get_tracer(ctx: TracingCtx) -> trace_api.Tracer:
    return ctx.provider.get_tracer("blob-upload-demo")


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def attributes_size_bytes(attributes: Mapping[str, Any]) -> int:
    """Approximate wire size of a span's attributes (keys + stringified values)."""
    return sum(len(k) + len(str(v)) for k, v in attributes.items())


def format_bytes(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


def print_size_table(rows: list[tuple[str, int, str]]) -> None:
    """rows: (label, attribute_bytes, note)"""
    label_width = max(len(label) for label, _, _ in rows) + 2
    print(f"\n  {'span':<{label_width}}{'attr bytes':>12}  note")
    print(f"  {'-' * label_width}{'-' * 12}  {'-' * 40}")
    for label, size, note in rows:
        print(f"  {label:<{label_width}}{format_bytes(size):>12}  {note}")


@dataclass
class Checker:
    failures: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        suffix = f" — {detail}" if detail else ""
        print(f"  [{status}] {name}{suffix}")
        if not ok:
            self.failures.append(name)

    def exit_code(self) -> int:
        if self.failures:
            print(f"\n{len(self.failures)} check(s) FAILED: {', '.join(self.failures)}")
            return 1
        print("\nAll checks passed.")
        return 0


def fetch_url(url: str, timeout_s: float = 5.0) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout_s) as resp:  # noqa: S310
        return resp.read()  # type: ignore[no-any-return]


def maybe_wait_for_browsing(uploader: LocalHttpBlobUploader) -> None:
    """Keep the blob HTTP server alive so Phoenix can render the URIs.

    Skipped when stdin is not a TTY (CI / scripted runs) or --no-wait is set.
    """
    if "--no-wait" in sys.argv or not sys.stdin.isatty():
        print("[blob] exiting immediately (--no-wait or non-interactive run);")
        print("[blob] re-run a demo script to serve blob URIs for the Phoenix UI.")
        return
    try:
        input(
            "\nBlob server is still serving — open Phoenix now to inspect the spans.\n"
            "Press Enter (or Ctrl-C) to stop the blob server and exit... "
        )
    except (KeyboardInterrupt, EOFError):
        pass
    uploader.stop_serving()
