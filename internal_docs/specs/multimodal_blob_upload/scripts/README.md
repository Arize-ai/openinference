# Blob-upload demo scripts

A runnable demo that **proves** the experimental blob-upload design captures large
multimodal content end-to-end: base64 audio goes into an OpenAI chat completion, the
instrumentation externalizes it at capture time, and the finished span carries only
storage URIs — in both the OpenInference attributes and the OTel GenAI dual-write.

The script:

1. Generates small, real (playable) WAV clips and sends one as a chat-completions
   `input_audio` content part; a mocked transport returns an audio response with a
   transcript (`gpt-4o-audio-preview` response shape).
2. Runs the real `OpenAIInstrumentor` with a `TraceConfig` whose
   `base64_media_max_length` forces externalization, once per uploader:
   - **Part 1** — the built-in `FsspecBlobUploader` writing to a local directory
     (content-addressed `{sha256}.wav` destinations, background bounded queue).
   - **Part 2** — a custom `ManifestBlobUploader` implementing the `BlobUploader`
     protocol from scratch (writes files plus a `manifest.json` sidecar recording
     mime type, modality, sha256, and the span attribute each blob came from).
3. Prints every span from an `InMemorySpanExporter`: the
   `…message_content.audio.audio.url` attributes hold storage URIs instead of base64,
   `audio.mime_type` / `audio.transcript` ride alongside, and with
   `enable_genai_semconv=True` the `gen_ai.input.messages` / `gen_ai.output.messages`
   attributes show the externalized audio as spec-conformant `uri` message parts.
4. Leaves the uploaded `.wav` files (and Part 2's `manifest.json`) on disk under a
   temp directory so you can verify the bytes round-tripped and are playable.

What makes it a real proof: the mock response's audio payload only exists inside the
HTTP response body. If capture, data-URI normalization, threshold detection, upload,
or URI substitution failed anywhere, the printed span would show base64, `__REDACTED__`,
or nothing — not a resolvable file URI whose content hash matches the uploaded file.

## Prerequisites

None. The OpenAI HTTP layer is mocked with `httpx.MockTransport` (no API key, no
network), local blob destinations need no fsspec install, and the script's inline
metadata (`[tool.uv.sources]`) points at this repo's editable packages so it always
exercises the in-branch implementation.

## Run

```bash
uv run --script internal_docs/specs/multimodal_blob_upload/scripts/audio_blob_upload_demo.py
```

Captured output from a full run is committed alongside as
[`audio_blob_upload_demo.txt`](./audio_blob_upload_demo.txt).

## What to look for in the output

| Output section | What it proves |
|---|---|
| `…audio.audio.url = file:///…/<sha256>.wav` (input and output messages) | Base64 in both directions was externalized; the URI is content-addressed and known before the background write completes. |
| `input.value = {…"data": "/…/<sha256>.wav"…}` | The raw-request JSON blob got the same treatment via `redact_media_from_request_parameters` — no base64 hiding inside `input.value`. |
| `gen_ai.input.messages` → `{"type": "uri", "modality": "audio", …}` | The GenAI dual-write emits the semconv `UriPart` for externalized media (a data URI would have been a `blob` part). |
| `files under …/fsspec-upload` listing | The decoded bytes actually landed on disk (8,044-byte playable WAVs), deduplicated by content hash. |
| Part 2 `manifest.json` | The custom hook received full provenance on each `Blob`: mime type, modality, sha256, and the originating span attribute key. |

## Configuration exercised

| Setting | Value in the demo | Notes |
|---|---|---|
| `blob_uploader` | `FsspecBlobUploader(base_path=<dir>)` / custom `ManifestBlobUploader` | zero-code equivalent: `OPENINFERENCE_BLOB_UPLOAD_BASE_PATH` |
| `base64_media_max_length` | `1_000` | demo threshold so a ~10 KB data URI triggers upload; default is `32_000` |
| `enable_genai_semconv` | `True` (Part 1) | shows the `uri` part in `gen_ai.*` messages |

## Follow-ups

- **openai-agents demo.** The realtime module (`openinference-instrumentation-openai-agents`)
  still uses its own audio env vars and `input.audio.url` attribute names; once migrated to
  the core config, add a sibling demo covering realtime PCM buffers.
- **PDF / `input_file` demo.** The same machinery handles `file` parts (chat completions)
  and `input_file` (Responses API); a document-modality sibling script would exercise
  `file.url` / `file.id` capture and `modality: "document"` conversion.
- **Remote-store demo.** Point `FsspecBlobUploader` at `s3://`/`gs://` (requires
  `openinference-instrumentation[blob-upload]` plus the store's fsspec driver) and confirm
  consumer-side URI resolution expectations.

## Layout

```
scripts/
├── README.md
├── audio_blob_upload_demo.py   — offline chat-completions audio demo (uv-runnable)
└── audio_blob_upload_demo.txt  — captured output from a full run
```
