# Blob-upload before/after demo scripts

Two self-contained scripts that **prove** the [blob-upload design](../blob_uploads.md):
large multimodal content that today rides on spans as (truncated) base64 data URIs is
instead handed to the proposed `BlobUploader` at capture time, and the span attribute
records only the destination URI — with the bytes fully recoverable from the store.

No API keys and no LLM calls are needed: the audio script synthesizes PCM16 in the
exact openai-agents realtime wire form (PR #3173), and the image script generates a
PNG locally and drives it through the *released* `OITracer`/`TraceConfig` masking
pipeline.

| script | before | after |
|---|---|---|
| `audio_blob_demo.py` | `input.audio.url` / `output.audio.url` = `data:audio/wav;base64,...` truncated at 32,000 chars (~0.5 s survives, invalid WAV) | same keys = `http://127.0.0.1:8321/<sha>.wav`; full WAV round-trips byte-for-byte |
| `image_blob_demo.py` | `message_content.image.image.url` = `__REDACTED__` (default config) or an ~884 KB inline base64 attribute (raised limit) | same key = `http://127.0.0.1:8321/<sha>.png`; PNG round-trips and renders in Phoenix |

Both scripts print PASS/FAIL per assertion and exit non-zero on failure.

## Prerequisites

```bash
# 1. Phoenix locally
pip install arize-phoenix
phoenix serve   # listens on http://localhost:6006

# 2. (optional) override endpoints
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export BLOB_DEMO_PORT=8321   # port for the demo blob store's HTTP server
```

Only [`uv`](https://docs.astral.sh/uv/) is otherwise required — dependencies are PEP 723
inline metadata, pinned to released packages (`openinference-instrumentation==0.1.54`,
`openinference-semantic-conventions==0.1.30`).

## Run

```bash
uv run --script internal_docs/specs/blob_uploads/scripts/audio_blob_demo.py
uv run --script internal_docs/specs/blob_uploads/scripts/image_blob_demo.py
```

When run interactively, each script keeps its blob HTTP server alive after the PASS
summary (press Enter to exit) so the Phoenix UI can resolve the blob URIs while you
browse. Pass `--no-wait` (or pipe stdin) for CI-style runs; re-running either script
re-serves everything in `blob_store/`.

The demo blob store is a local directory (`scripts/blob_store/`, gitignored) served at
`http://127.0.0.1:8321` — the stand-in for S3/GCS. Objects are content-addressed
(`sha256[:20]` + mime extension), so identical payloads dedup across runs, and writes
happen on a background worker fed by a bounded queue, exactly the async model the spec
proposes.

## What to look at in Phoenix (http://localhost:6006)

### Project `blob-upload-audio-demo` (audio path)

Two traces, each a PR #3173-shaped realtime turn:
`conversation.turn` (AUDIO) → `user` (USER) + `assistant` (LLM).

1. Open **`conversation.turn — before (inline base64)`** → `user` span → attributes
   pane: `input.audio.url` is a `data:audio/wav;base64,...` string that ends abruptly —
   32,000 chars, ~0.5 s of the 3.2 s utterance, not decodable as WAV. Same for
   `output.audio.url` on the `assistant` span.
2. Open **`conversation.turn — after (blob upload)`**: the same attributes are now
   short `http://127.0.0.1:8321/<sha>.wav` URIs. Click/copy one into the browser while
   the script is serving — the full WAV downloads and plays. `input.audio.transcript`
   and `input.audio.mime_type` are identical in both traces: only the payload moved.
3. Note the span sizes: ~32 KB of attributes per audio span before, a few hundred
   bytes after (the console prints the exact table).

### Project `blob-upload-image-demo` (image path)

Three LLM chat spans written by **identical code** — only the `TraceConfig` differs:

1. **`before — default config (image redacted)`**: open the span's input messages —
   the image part shows `__REDACTED__`. This is released behavior for any input image
   whose base64 exceeds 32,000 chars: the content is destroyed.
2. **`before — raised limit (inline base64)`**: the image renders inline (Phoenix
   displays data-URI images), but the attributes pane shows the cost — an ~884 KB
   attribute on one span.
3. **`after — blob upload (external URI)`**: the image renders identically (Phoenix's
   `<SpanImage>` takes any URL) while the attribute is a 46-char URI. The user text
   part sits inline next to it, untouched.

Audio caveat: Phoenix has no audio player in span details yet, so for the audio project
the evidence is the attribute value itself (and fetching the URI); images render
in-page for the full effect.

## Layout

```
scripts/
├── README.md
├── common.py            — proposed Blob/BlobUploader interface (mirrors ../blob_uploads.md §3.1),
│                          LocalHttpBlobUploader demo backend, WAV/data-URI helpers copied from
│                          _realtime.py, OTel→Phoenix setup, PASS/FAIL + size-table helpers
├── audio_blob_demo.py   — realtime turn wire form; integration point 2 (direct capture-time call)
├── image_blob_demo.py   — multimodal chat span; integration point 1 (TraceConfig choke point,
│                          via a BlobOffloadingTraceConfig subclass of the released TraceConfig)
└── blob_store/          — content-addressed demo storage (gitignored, created on first run)
```

## Knobs to experiment with

The audio demo implements the techspec's offload policy (§3.4) and its PASS checks
adapt, so the policy matrix can be explored directly:

```bash
# raise the inline budget above the payload size: audio stays inline on BOTH
# variants — content that fits inline is never offloaded
OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH=10000000 uv run --script .../audio_blob_demo.py

# privacy wins over upload: no input.audio.* attribute is written and no blob
# is uploaded, even with an uploader configured
OPENINFERENCE_HIDE_INPUT_AUDIO=1 uv run --script .../audio_blob_demo.py
OPENINFERENCE_HIDE_OUTPUT_AUDIO=1 uv run --script .../audio_blob_demo.py
```
