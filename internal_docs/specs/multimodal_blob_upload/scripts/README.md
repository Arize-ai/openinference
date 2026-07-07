# Blob-upload demo scripts

A live-API demo that **proves** the experimental blob-upload design captures large
multimodal content end-to-end: a base64 PNG goes into a real OpenAI vision chat
completion, the instrumentation externalizes it at capture time, and the finished span
carries only storage URIs — in both the OpenInference attributes and the OTel GenAI
dual-write. Spans are exported to a local Phoenix for inspection in the UI.

The script:

1. Generates a real PNG in pure Python (a red circle on white — something the model
   can actually describe) and sends it as a base64 `image_url` data URI to the live
   OpenAI API (`gpt-4o-mini` by default; override with `OPENAI_MODEL`).
2. Runs the real `OpenAIInstrumentor` with a `TraceConfig` whose
   `base64_image_max_length=1_000` forces externalization of the ~2.3k-char data URI,
   once per scenario:
   - **Scenario A** — the built-in `FsspecBlobUploader` writing to a local directory
     (content-addressed `{sha256}.png` destinations, background bounded queue), with
     `enable_genai_semconv=True` so the dual-write is visible.
   - **Scenario B** — a custom `ManifestBlobUploader` implementing the `BlobUploader`
     protocol from scratch (writes files plus a `manifest.json` sidecar recording mime
     type, modality, sha256, and the span attribute each blob came from).
3. Captures spans with an `InMemorySpanExporter`, prints them, exports the same spans
   to Phoenix (`{PHOENIX_COLLECTOR_ENDPOINT}/v1/traces`, project
   `blob-upload-image-demo`), and runs PASS/FAIL assertions. Exits non-zero on failure.

What makes it a real proof: the PNG bytes exist only in this process. If capture,
threshold detection, upload, or URI substitution failed anywhere, the span would carry
base64 or `__REDACTED__` — and the assertion comparing the uploaded file's sha256
against the original PNG would fail.

## Prerequisites

```bash
# 1. Phoenix locally
pip install arize-phoenix
phoenix serve   # listens on http://localhost:6006

# 2. API key
export OPENAI_API_KEY=...

# 3. (optional) overrides
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export OPENAI_MODEL=gpt-4o-mini
export OPENINFERENCE_BLOB_UPLOAD_BASE_PATH=/somewhere/persistent  # default: a temp dir
```

The script's inline metadata (`[tool.uv.sources]`) points at this repo's editable
packages, so it always exercises the in-branch implementation.

## Run

```bash
uv run --script internal_docs/specs/multimodal_blob_upload/scripts/openai_image_blob_upload.py
```

Each scenario prints the model's answer, the span attributes, and PASS/FAIL per
assertion; the script exits non-zero on any failure. To refresh the committed output
log after a run:

```bash
uv run --script internal_docs/specs/multimodal_blob_upload/scripts/openai_image_blob_upload.py \
  | tee internal_docs/specs/multimodal_blob_upload/scripts/openai_image_blob_upload.txt
```

Open the Phoenix UI at <http://localhost:6006> and select the
`blob-upload-image-demo` project to inspect the spans.

## Scenarios and assertions

| Scenario | What it proves |
|---|---|
| **A. FsspecBlobUploader** | The built-in local/fsspec path: the image attribute holds a content-addressed URI, the uploaded file's sha256 matches the original PNG, and `gen_ai.input.messages` carries the image as a `uri` part (not a `blob` part). |
| **B. custom ManifestBlobUploader** | The `BlobUploader` protocol is sufficient for a from-scratch hook: same assertions, plus the manifest shows the hook received full provenance (mime type, modality, sha256, originating span attribute key). |

Assertions per scenario:

- `…message_content.image.image.url` holds a URI — not a `data:` URI, not `__REDACTED__`.
- The file at that URI exists and its sha256 matches the original PNG.
- The file landed under the scenario's upload directory.
- (Scenario A) `gen_ai.input.messages` represents the image as `{"type": "uri", "modality": "image", …}`.

## What to look for in Phoenix

- The LLM span's input message shows the image content item whose URL is a storage
  URI instead of inline base64 — the span stays small no matter how large the image.
- `input.value` (the raw request JSON) shows the image redacted: images inside
  `input.value` currently redact rather than externalize — a documented follow-up in
  the techspec (§10) — while audio/file payloads inside `input.value` do externalize.
- Note: the Phoenix UI renders `http(s)`/`data:` image URLs in the span details view;
  `file://` (and `s3://`/`gs://`) URIs are recorded faithfully but not rendered —
  consumer-side URI resolution is out of scope for this branch (techspec §10).

## Follow-ups

- **Audio / PDF siblings.** Audio (`input_audio`, chat-completions audio responses) and
  documents (`file` / `input_file` parts) run through the same machinery with
  `base64_media_max_length`; an earlier offline audio variant of this demo lives in git
  history (`scripts/audio_blob_upload_demo.py`, removed when this script replaced it).
- **openai-agents realtime demo.** Once the realtime module migrates from its private
  audio env vars onto the core config, add a sibling demo covering realtime PCM buffers.
- **Remote-store demo.** Point `FsspecBlobUploader` at `s3://`/`gs://` (requires
  `openinference-instrumentation[blob-upload]` plus the store's fsspec driver) and
  confirm consumer-side URI resolution expectations.

## Layout

```
scripts/
├── README.md
├── openai_image_blob_upload.py   — live-API image demo, exports to Phoenix (uv-runnable)
└── openai_image_blob_upload.txt  — captured output from a real run (regenerate via tee)
```
