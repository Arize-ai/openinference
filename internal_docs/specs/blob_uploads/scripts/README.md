# Blob-upload demo scripts

Two live demos of the [blob-upload design](../blob_uploads.md), each driving a **real
instrumented OpenAI Agents SDK app** — no hand-built spans. Large multimodal content
that today rides on spans as truncated or redacted base64 is handed to the proposed
`BlobUploader` at capture time; the span attribute records only the destination URI.
Each script prints the resulting spans and exports them to Phoenix.

| script | the app | what it shows |
|---|---|---|
| `image_blob_demo.py` | a vision agent (`Agent` + `Runner.run` with an `input_image`), instrumented by openinference-instrumentation-openai-agents **and** openinference-instrumentation-openai (the Responses API call underneath) | the same run twice, changing only the `TraceConfig`: `message_content.image.image.url` = `__REDACTED__` (default) vs a blob-store path (blob-upload config) |
| `audio_blob_demo.py` | a **live Realtime API session** (`RealtimeAgent` + `RealtimeRunner`): a TTS-spoken question goes in, the assistant answers in audio, the released realtime instrumentation produces the AUDIO/USER/LLM span tree | with the capture site patched per techspec §2.3, `input.audio.url` / `output.audio.url` are blob-store paths and the full WAVs survive; with `--inline`, today's released behavior (data URIs truncated at 32,000 chars) |

The demo store (`LocalBlobStore` in `common.py`) is deliberately simple: it writes
content-addressed files under `scripts/blob_store/` (gitignored) and returns the
file's **repo-root-relative path** as the URI, e.g.
`internal_docs/specs/blob_uploads/scripts/blob_store/3a7bd3….wav`. Phoenix displays
the URI as an ordinary string attribute — resolving or rendering it is the backend's
responsibility. `common.py` holds only the proposed `Blob`/`BlobUploader` interface
(the pieces that would move into `openinference-instrumentation`) plus this mock.

## Prerequisites

```bash
# 1. Phoenix locally
pip install arize-phoenix
phoenix serve                    # http://localhost:6006

# 2. OpenAI API key (both scripts make real API calls;
#    the audio demo needs Realtime API access)
export OPENAI_API_KEY=...

# 3. (optional) overrides
export PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006
export OPENAI_MODEL=gpt-4o-mini          # vision model (image demo)
export OPENAI_TTS_MODEL=gpt-4o-mini-tts  # TTS voice for the spoken question (audio demo)
export OPENAI_REALTIME_MODEL=gpt-realtime
export REALTIME_DEMO_TIMEOUT_SEC=120     # hard deadline on the realtime session (audio demo)
```

Only [`uv`](https://docs.astral.sh/uv/) is otherwise required — dependencies are PEP 723
inline metadata resolved into ephemeral environments.

## Run

```bash
uv run --script internal_docs/specs/blob_uploads/scripts/image_blob_demo.py
uv run --script internal_docs/specs/blob_uploads/scripts/audio_blob_demo.py
uv run --script internal_docs/specs/blob_uploads/scripts/audio_blob_demo.py --inline  # today's behavior
```

Each script prints the spans it produced (attribute by attribute, long values elided
with their true size) and exits; the blobs stay under `scripts/blob_store/`.

## What to look at in Phoenix (http://localhost:6006)

**Project `blob-upload-image-demo`** — two agent traces from identical app code
(`Agent workflow` → `Vision Assistant` → `turn` → `response` → `Response`; the image
attribute lives on the innermost `Response` LLM span):

1. First run (default config): the `Response` span's input messages show the image
   part as `__REDACTED__` — today's released behavior for any input image whose
   base64 exceeds 32,000 chars (the only alternative today is raising the budget and
   carrying ~884 KB of base64 on the span).
2. Second run (blob-upload config): the same attribute holds
   `internal_docs/…/blob_store/<sha>.png` — a short path instead of a redaction
   marker; the bytes are in that file, deduped by content hash.
3. `input.value` is small in both — the instrumentor's existing pre-pass strips the
   base64 image from the serialized request. Upgrading that redaction to a blob URI
   is open question 1 in the techspec.

**Project `blob-upload-audio-demo`** — one realtime trace per run:
`conversation.turn` (AUDIO) → `user` (USER) + `assistant` (LLM), produced by the
released realtime instrumentation from a live session (real whisper transcripts,
token counts, time-to-first-token):

1. Default run (capture site patched): `input.audio.url` / `output.audio.url` are
   blob-store paths; the WAV files under `scripts/blob_store/` contain the full
   spoken question and answer — play them.
2. `--inline` run: the same attributes are `data:audio/wav;base64,…` cut off at
   32,000 chars (~0.5 s of audio, not decodable) — the span carries ~32 KB per side
   and the content is still lost.

## Layout

```
scripts/
├── README.md
├── common.py            — proposed Blob/BlobUploader interface (../blob_uploads.md §2.1)
│                          + LocalBlobStore, the repo-local mock store
├── image_blob_demo.py   — integration point 1 (§2.2): TraceConfig choke point, driven by a
│                          real Agents SDK vision run
├── audio_blob_demo.py   — integration point 2 (§2.3): capture-site upload, driven by a
│                          live Realtime API session
└── blob_store/          — content-addressed demo storage (gitignored, created on first run)
```
