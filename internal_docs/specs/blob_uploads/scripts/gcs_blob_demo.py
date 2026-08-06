# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-cloud-storage>=2.14",
#     "openai>=1.60.0",
#     "openinference-instrumentation-openai>=0.1.52",
#     "openinference-instrumentation",
#     "opentelemetry-sdk>=1.42.0",
#     "opentelemetry-exporter-otlp-proto-http>=1.42.0",
#     "pillow>=10.0.0",
# ]
#
# [tool.uv.sources]
# openinference-instrumentation = { path = "../../../../python/openinference-instrumentation" }
# ///
"""Blob upload to a real Google Cloud Storage bucket — redact vs inline vs offload.

Edit ``BASE64_IMAGE_MAX_LENGTH`` below (default 32,000) — that is the
``TraceConfig.base64_image_max_length`` budget. The same vision request runs
twice with that value:

  run 1 — no uploader        over-cutoff image → ``__REDACTED__``
  run 2 — GCS uploader       over-cutoff image → an ``https://`` (or signed) URL

The generated image is ~880,000 chars as a data URI, so the default shows
**redact** then **offload**. Raise the constant above that length (e.g.
``1_000_000``) to see **inline** data URIs in both runs.

The recorded URI is the most viewer-resolvable form available:

  1. a V4 **signed URL** (publicly fetchable until it expires,
     ``GCS_SIGNED_URL_HOURS``) when the credentials can sign — a
     service-account key, or user credentials plus
     ``GCS_SIGNING_SERVICE_ACCOUNT=<sa-email>`` (requires Token Creator on
     that SA; signing goes through IAM signBlob);
  2. otherwise ``https://storage.googleapis.com/<bucket>/<object>`` — a valid
     URL that resolves for viewers with bucket access (or public buckets).

Requires: OPENAI_API_KEY, GCS_BUCKET (+ optional GCS_PREFIX), and
GOOGLE_CLOUD_PROJECT if your application-default credentials carry no quota
project. Uses ADC (``gcloud auth application-default login``). Start Phoenix
locally (``phoenix serve``) before running.

Run:
  GCS_BUCKET=my-bucket GCS_PREFIX=me/blobs \\
    uv run --script internal_docs/specs/blob_uploads/scripts/gcs_blob_demo.py
"""

from __future__ import annotations

import atexit
import base64
import datetime
import os
import random
import sys
from io import BytesIO
from typing import Optional

# Ignore ~/.netrc for this process. Some machines have an entry that makes
# HTTP clients send Basic auth to storage.googleapis.com, which GCS rejects
# (401 "HTTP Basic Authentication is not supported") and the span falls back
# to __REDACTED__. Override with NETRC=/path/to/netrc if you need one.
os.environ.setdefault("NETRC", os.devnull)

from openai import OpenAI
from openinference.instrumentation import Blob, BlobUploader, TraceConfig
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from PIL import Image

PROJECT_NAME = "blob-upload-gcs-demo"

# ``TraceConfig.base64_image_max_length``. Default matches today's released
# budget. Edit this to try other outcomes:
#   - leave at 32_000 → run 1 redacts, run 2 offloads (an https:// or signed URL)
#   - set above the printed data-URI length (e.g. 1_000_000) → both runs inline
BASE64_IMAGE_MAX_LENGTH = 32_000


class GcsBlobUploader(BlobUploader):
    """A ``BlobUploader`` writing content-addressed objects to a GCS bucket.

    Subclassing ``BlobUploader`` is optional (any object with matching
    ``upload``/``shutdown`` methods satisfies the protocol) but makes the
    contract explicit. Synchronous for demo clarity — a production
    implementation moves bytes on a background worker and returns the
    (precomputable) URI immediately.
    """

    def __init__(self, bucket: str, prefix: str) -> None:
        from google.cloud import storage

        client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
        self._bucket = client.bucket(bucket)
        self._prefix = prefix.strip("/")
        self._signer = _signing_credentials(client)
        self._signed_url_hours = float(os.environ.get("GCS_SIGNED_URL_HOURS", "24"))
        if self._signer is None:
            print(
                "[gcs] no signing credentials — recording plain https URLs "
                "(set GCS_SIGNING_SERVICE_ACCOUNT or use a service-account "
                "key for publicly fetchable signed URLs)"
            )

    def upload(self, blob: Blob) -> Optional[str]:
        ext = {"image/png": ".png", "image/jpeg": ".jpg"}.get(blob.mime_type, ".bin")
        name = f"{self._prefix}/{blob.content_sha256[:20]}{ext}"
        obj = self._bucket.blob(name)
        if not obj.exists():  # content-addressed dedup
            obj.upload_from_string(blob.data, content_type=blob.mime_type)
            print(
                f"[gcs] uploaded {len(blob.data):,} B → gs://{self._bucket.name}/{name}"
            )
        if self._signer is not None:
            return str(
                obj.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(hours=self._signed_url_hours),
                    credentials=self._signer,
                )
            )
        return f"https://storage.googleapis.com/{self._bucket.name}/{name}"

    def shutdown(self, timeout_sec: float = 10.0) -> None:
        pass  # synchronous demo — nothing pending


def _signing_credentials(client: "object") -> Optional[object]:
    """Signed URLs need a signer: a service-account key signs natively; user
    credentials can sign via IAM signBlob when impersonating a service
    account (GCS_SIGNING_SERVICE_ACCOUNT) they hold Token Creator on."""
    from google.auth import credentials as ga_credentials

    source = client._credentials  # type: ignore[attr-defined]
    if sa_email := os.environ.get("GCS_SIGNING_SERVICE_ACCOUNT"):
        from google.auth import impersonated_credentials

        return impersonated_credentials.Credentials(
            source_credentials=source,
            target_principal=sa_email,
            target_scopes=["https://www.googleapis.com/auth/devstorage.read_write"],
        )
    if isinstance(source, ga_credentials.Signing):
        return source
    return None


def make_demo_png() -> bytes:
    """Seeded RGB noise — incompressible, so the PNG is realistically large."""
    rng = random.Random(42)
    img = Image.frombytes("RGB", (640, 400), rng.randbytes(640 * 400 * 3))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def ask_about_image(data_uri: str) -> str:
    response = OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "low"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set — this demo makes real vision calls.")
    gcs_bucket = os.environ.get("GCS_BUCKET")
    if not gcs_bucket:
        sys.exit(
            "GCS_BUCKET is not set — e.g. GCS_BUCKET=my-bucket GCS_PREFIX=me/blobs"
        )

    png = make_demo_png()
    data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    print(f"image: {len(png):,} B PNG → {len(data_uri):,} chars as a data URI")
    print(f"BASE64_IMAGE_MAX_LENGTH: {BASE64_IMAGE_MAX_LENGTH:,}")

    phoenix = os.environ.get(
        "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
    ).rstrip("/")
    provider = TracerProvider(
        resource=Resource.create({"openinference.project.name": PROJECT_NAME})
    )
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(f"{phoenix}/v1/traces"))
    )

    uploader = GcsBlobUploader(gcs_bucket, os.environ.get("GCS_PREFIX", "oi-media"))
    # Implementations own their exit flush (this one is synchronous, but the
    # pattern matters for real background uploaders).
    atexit.register(uploader.shutdown)
    for label, config in [
        (
            "run 1: no uploader",
            TraceConfig(base64_image_max_length=BASE64_IMAGE_MAX_LENGTH),
        ),
        (
            "run 2: GCS uploader",
            TraceConfig(
                base64_image_max_length=BASE64_IMAGE_MAX_LENGTH,
                blob_uploader=uploader,
            ),
        ),
    ]:
        OpenAIInstrumentor().instrument(tracer_provider=provider, config=config)
        answer = ask_about_image(data_uri)
        OpenAIInstrumentor().uninstrument()
        print(f"\n[{label}] model: {answer}")
        provider.force_flush()

    provider.shutdown()
    print(
        f"\nCheck your local Phoenix at {phoenix} → project {PROJECT_NAME!r}.\n"
        "Compare the two ChatCompletion spans' message_content.image.image.url:\n"
        "  run 1 → __REDACTED__ (or a data: URI if you raised BASE64_IMAGE_MAX_LENGTH)\n"
        "  run 2 → an https:// (or signed) URL (or a data: URI if under the cutoff)\n"
        "Edit BASE64_IMAGE_MAX_LENGTH in this file and re-run to try other outcomes."
    )


if __name__ == "__main__":
    main()
