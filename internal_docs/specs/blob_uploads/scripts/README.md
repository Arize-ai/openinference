# Blob-upload demo scripts

A live demo of the [blob-upload design](../blob_uploads.md), driving a **real
auto-instrumented OpenAI vision call** — no hand-built spans. An oversized base64
image that today rides on the span as `__REDACTED__` is handed to a `BlobUploader` at
capture time; the span attribute records only the destination URI. The script imports
the **live** `openinference-instrumentation` package from this repo (a
`[tool.uv.sources]` path in its PEP 723 metadata), so it exercises the shipped
`Blob`/`BlobUploader`/`TraceConfig` code — no vendored copies. It prints the resulting
spans and exports them to Phoenix.

`image_blob_demo.py` makes one chat-completions vision request (text + a ~600 KB
base64 PNG), instrumented by openinference-instrumentation-openai. The same request
runs twice, changing only the `TraceConfig`: `message_content.image.image.url` =
`__REDACTED__` (default) vs a locally served URL (`TraceConfig(blob_uploader=...)`).

`gcs_blob_demo.py` is the same vision call against a **real Google Cloud Storage
bucket**. Edit the `BASE64_IMAGE_MAX_LENGTH` constant in the script (default
32,000) — that is the `base64_image_max_length` budget. Each execution runs twice
with that value — run 1 without an uploader, run 2 with `GcsBlobUploader` — so an
over-cutoff image shows **redact** then **offload** (an `https://` or signed URL);
raise the constant above the printed data-URI length (e.g. `1_000_000`) to see
**inline** in both. Check the two spans in local Phoenix
(project `blob-upload-gcs-demo`). It authenticates via application-default
credentials (`gcloud auth application-default login`); set `GCS_BUCKET`, optionally
`GCS_PREFIX`, and `GOOGLE_CLOUD_PROJECT` if your ADC has no quota project.

The demo store (`LocalBlobStore`, inlined in the image script and explicitly
subclassing `BlobUploader`) writes content-addressed files under
`scripts/blob_store/` (gitignored) and serves them from a background HTTP thread,
recording a real `http://localhost:<port>/<sha>.png` URL — Phoenix renders the image
inline while the script keeps serving (it waits for Enter before exiting).
OpenInference ships no uploader implementation, which is exactly why the demos carry
their own.

The GCS demo records the most viewer-resolvable URL its credentials allow: a V4
**signed URL** (publicly fetchable until expiry, `GCS_SIGNED_URL_HOURS`) when the
credentials can sign — a service-account key, or user credentials plus
`GCS_SIGNING_SERVICE_ACCOUNT=<sa-email>` (IAM signBlob impersonation, requires Token
Creator on that SA) — otherwise the plain
`https://storage.googleapis.com/<bucket>/<object>` form, which resolves for viewers
with bucket access.

## Prerequisites

```bash
# 1. Phoenix locally
pip install arize-phoenix
phoenix serve                    # http://localhost:6006

# 2. OpenAI API key (the script makes real vision calls)
export OPENAI_API_KEY=...

# 3. (optional) overrides
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export OPENAI_MODEL=gpt-4o-mini          # vision model
```

Only [`uv`](https://docs.astral.sh/uv/) is otherwise required — dependencies are PEP 723
inline metadata resolved into an ephemeral environment.

## Run

```bash
uv run --script internal_docs/specs/blob_uploads/scripts/image_blob_demo.py
```

```bash
GCS_BUCKET=my-bucket GCS_PREFIX=me/blobs \
  uv run --script internal_docs/specs/blob_uploads/scripts/gcs_blob_demo.py
```

Edit `BASE64_IMAGE_MAX_LENGTH` in `gcs_blob_demo.py` to try redact / inline / offload.

The script prints the spans it produced (attribute by attribute, long values elided
with their true size) and exits; the blobs stay under `scripts/blob_store/`.

## What to look at in Phoenix (http://localhost:6006)

**Project `blob-upload-image-demo`** — one `ChatCompletion` LLM span per run, from
identical app code:

1. First run (default config): the span's input messages show the image part as
   `__REDACTED__` — today's released behavior for any input image whose base64
   exceeds 32,000 chars (the only alternative today is raising the budget and
   carrying ~884 KB of base64 on the span).
2. Second run (blob-upload config): the same attribute holds
   `http://localhost:<port>/<sha>.png` — a real URL Phoenix renders inline while the
   script keeps serving; the bytes are deduped by content hash.
3. `input.value` is small in both runs — the instrumentor's existing pre-pass strips
   the base64 image from the serialized request. Upgrading that redaction to a blob
   URI is future work (step 5 of the techspec's
   [media-type checklist](../blob_uploads.md#6-checklist-adding-offload-support-for-a-new-media-type)).

## Layout

```
scripts/
├── README.md
├── image_blob_demo.py   — the live TraceConfig(blob_uploader=...) mask() choke point
│                          (the techspec's shipped integration point), driven by a
│                          real auto-instrumented OpenAI vision call; local blob store
├── gcs_blob_demo.py     — same call against a real GCS bucket; edit
│                          BASE64_IMAGE_MAX_LENGTH to show redact / inline / offload
└── blob_store/          — content-addressed demo storage (gitignored, created on first run)
```
