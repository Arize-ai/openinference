# Blob-upload demo script

A short, self-contained demo of **how a custom blob-upload hook works**: a real OpenAI
vision call is made with an inline base64 PNG; instead of recording the base64 in the
span, the instrumentation hands the decoded bytes to a custom `BlobUploader` (a ~20-line
class implementing the two-method protocol) and records the URI it returns. The span is
printed and exported to Phoenix.

The educational core is the `MyBlobUploader` class at the top of the script — its
docstring states the protocol contract (return the URI immediately, write out-of-band,
return `None` to reject) and what the `Blob` argument carries (decoded bytes, mime type,
modality, sha256, originating span attribute key). Swap its body for S3/GCS or an
artifact store; the built-in `FsspecBlobUploader` is the batteries-included version.

## Prerequisites

```bash
# 1. Phoenix locally (optional — the script still prints the span without it)
pip install arize-phoenix
phoenix serve   # listens on http://localhost:6006

# 2. API key
export OPENAI_API_KEY=...

# 3. (optional) overrides
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export OPENAI_MODEL=gpt-4o-mini
```

The script's inline metadata (`[tool.uv.sources]`) points at this repo's editable
packages, so it always exercises the in-branch implementation.

## Run

```bash
uv run --script internal_docs/specs/multimodal_blob_upload/scripts/openai_image_blob_upload.py
```

## What to look for

- `uploaded image/png (… bytes) from llm.input_messages.0.…image.image.url` — the hook
  received the decoded bytes with full provenance.
- `…message_content.image.image.url = file:///…/<sha>.png` — the span holds a URI, not
  base64 (the demo threshold is `base64_image_max_length=1_000`; the default is 32,000).
- `gen_ai.input.messages` contains `{"type": "uri", "modality": "image", …}` — the OTel
  GenAI dual-write emits the semconv `UriPart` for externalized media.
- In the Phoenix UI (project `default`): the span stays small no matter how large the
  image. Note that `input.value` shows the image redacted (images inside `input.value`
  redact rather than externalize — techspec §10 follow-up), and Phoenix renders
  `http(s)`/`data:` image URLs but not `file://` URIs — consumer-side URI resolution is
  out of scope for this branch.

## Layout

```
scripts/
├── README.md
└── openai_image_blob_upload.py   — custom-uploader demo, live API + Phoenix (uv-runnable)
```
