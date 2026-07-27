# Blob upload for large multimodal span content

**Status:** accepted — interface-only scope per maintainer review (2026-07-20), narrowed
2026-07-27 to **image offload only**, the one media type whose message-content convention
is settled today; contract + policy land in
[#3409](https://github.com/Arize-ai/openinference/pull/3409). The audio/file vocabulary
([#3450](https://github.com/Arize-ai/openinference/pull/3450)) and OpenAI audio/file
capture ([#3410](https://github.com/Arize-ai/openinference/pull/3410)) are deferred until
those conventions land (open question 6); their branches are preserved.
**Demos:** [`scripts/`](./scripts/README.md) — live demos against a local Phoenix (requires `OPENAI_API_KEY`)
**Scope:** Python first; JS/Java noted as follow-on. Image offload ships; audio and file/PDF are design follow-ons.

## TL;DR

Large multimodal content (audio, images, PDFs) currently rides on spans as inline
`data:<mime>;base64,...` values that either get destroyed by size guards or blow up
span sizes. This design adds a pluggable **`BlobUploader`**: decoded bytes go to
external storage at capture time and the span attribute records only the destination
URI.

New public API in `openinference-instrumentation` — **interface and policy only, no
transport**: OI does not build or maintain upload machinery; implementations come from
applications, vendor SDKs (the Arize SDK will be the first), or a future upstream
(OTel util-genai) byte uploader.

```python
Blob(data, mime_type, modality="", attribute_key=None, content_sha256="")  # frozen dataclass
BlobUploader              # @runtime_checkable Protocol: upload(blob) -> Optional[str]; shutdown(timeout_sec)
load_blob_uploader(name)  # resolves the "openinference_blob_uploader" entry-point group
```

One new `TraceConfig` field (precedence code > env > default):

| field | env var | default |
|---|---|---|
| `blob_uploader` | `OPENINFERENCE_BLOB_UPLOADER=<name>` loads a registered `openinference_blob_uploader` entry point (mirrors util-genai's `opentelemetry_genai_completion_hook` mechanics) | `None` |

Everything else the policy needs already exists for images: `base64_image_max_length`
says *what* offloads, `hide_input_images` says what never leaves the process, and the
entry-point name says *who* moves the bytes (env-first, since instrumentation happens
automatically at the edge). The offloadable attribute set is built in — users never
enumerate fields. Audio/file budgets and hide flags arrive with their vocabulary
(open question 6).

One policy, applied inside the existing `TraceConfig.mask()` choke point (and by a
direct capture-time API for instrumentors that hold raw bytes):

> **hide** (never uploaded) → **fits inline budget** (stays a data URI) →
> **upload** (attribute = storage URI) → **`__REDACTED__`** (uniform fallback —
> no uploader, rejection, and failure all degrade to today's redaction, never block).

**No semconv additions.** Offload applies to the existing image message-content
convention only. Audio is emitted by a single instrumentor (openai-agents realtime)
under instrumentor-local keys, and files/PDF have no convention at all — introducing
that vocabulary is deferred (open question 6; the drafted constants, spec text, and
gen_ai mapping are preserved on the `blob-upload-semconv` branch, #3450). The `Blob`
contract itself stays media-agnostic — one `Blob` per media content part, §2.1 — so
the follow-ons add mask rules, not API.

The attribute key never changes; only the value does —
`...message_content.image.image.url` goes from `"__REDACTED__"` (or a multi-hundred-KB
data URI) to `"s3://my-bucket/oi-media/3a7bd3….png"`. With `enable_genai_semconv=True`
the dual-write already maps non-`data:` URLs to `uri` parts, so an externalized image
emits a spec-shaped part with **zero conversion changes**:
`{"type": "uri", "modality": "image", "uri": "s3://…"}`.

## 1. Problem

OpenInference spans capture multimodal content **inline** as `data:<mime>;base64,...`
attribute values. That worked for occasional small images; it does not survive contact
with production voice traces.

The forcing function is realtime audio tracing for openai-agents
([PR #3173](https://github.com/Arize-ai/openinference/pull/3173)). The OpenAI Realtime
API streams 24 kHz mono PCM16 both directions — 64 KB/s per side once base64-encoded.
The inline cap (default 32,000 chars) preserves roughly **half a second** of audio,
and the cut lands mid-stream, so the survivor is not even valid base64/WAV.
Production choices today:

| today's option | consequence |
|---|---|
| default truncation (audio) / redaction (image) | content destroyed — a 3.2 s question keeps ~0.5 s of unplayable audio; a >32 KB image becomes `__REDACTED__` |
| raise the max-length env vars | a 663 KB PNG becomes an 884 KB attribute; multi-MB spans stress OTLP payload limits (gRPC default rejects the whole 4 MB+ batch), collectors, and span stores |
| hide flags (`OPENINFERENCE_HIDE_INPUT_IMAGES`, …) | attribute never emitted — no observability at all |

The missing option, and the subject of this spec: **upload the decoded bytes to external
storage at capture time and record only the destination URI on the span**.

Realtime audio is the forcing function for the *contract* (non-blocking, a raw-bytes
door) — but the first shipped slice is **images**, because image is the only media
type whose message-content convention (`message_content.image.image.url` +
`base64_image_max_length` + `hide_input_images`) is settled and emitted by many
instrumentors today. Audio and file offload follow once their conventions land
(open question 6).

Two structural facts shape the design:

- Every attribute set on an `OITracer`-created span already flows through a single
  choke point, `TraceConfig.mask(key, value)`, where image redaction happens today —
  and where offload can happen with **zero instrumentor changes**.
- The realtime instrumentor holds **raw PCM bytes**, not data URIs, until the final
  encode. Forcing it through a data-URI round-trip just so the choke point can decode
  it again would double memory churn on a hot path, so it also needs a direct
  capture-time API.

## 2. Proposed design

### 2.1 Interface

New module in the core `openinference-instrumentation` package, public exports
`Blob`, `BlobUploader`, `load_blob_uploader`:

```python
@dataclass(frozen=True)
class Blob:
    data: bytes                          # decoded bytes — never base64 text
    mime_type: str                       # "audio/wav", "image/png", ...
    modality: str = ""                   # "image"|"audio"|"video"|"document";
                                         # derived from mime_type when omitted
    attribute_key: Optional[str] = None  # span attribute the ref lands on
    content_sha256: str = ""             # hex digest of data; computed automatically


@runtime_checkable
class BlobUploader(Protocol):
    def upload(self, blob: Blob) -> Optional[str]: ...
    def shutdown(self, timeout_sec: float = 10.0) -> None: ...
```

`modality` maps straight onto the gen_ai part `modality` field; `content_sha256` is
computed once so neither the caller nor the uploader re-hashes. Span/trace ids are
deliberately **not** part of `Blob`: content-addressed storage means the same payload
referenced from two traces is one object, so per-trace layouts don't compose with dedup.

Contract for `upload`:

- **One `Blob` per media content part; a single uploader serves every mime type.**
  There is no per-mime registry: `blob.mime_type` / `blob.modality` let an
  implementation route (e.g. per-modality prefixes) or refuse (return `None` → that
  part alone redacts; siblings unaffected) as its own policy.
- **MUST return quickly.** Compute the destination URI synchronously (content-hash
  naming); transfer bytes on a background worker. Capture sites sit on the realtime
  websocket event path.
- **`None` means "not uploaded — redact".** On backpressure, after shutdown, or by
  policy, the caller records `__REDACTED__` — the same value oversized content gets
  with no uploader at all.
- **Errors never propagate.** Worker-side failures are logged; the app is unaffected.
  Implementations SHOULD fail fast at construction when the destination is unusable
  (missing fsspec driver → `ImportError`; a startup write probe is recommended).
- **Content-hash naming is the default** (`{base_path}/{sha256(data)}.{ext}`, extension
  mapped from the mime type): identical payloads dedup — significant for multi-turn
  chats that resend the same image every turn — and the URI is computable before the
  bytes land.
- **`shutdown(timeout_sec)`** flushes the queue and stops workers; invoked from
  instrumentor `uninstrument()` and an `atexit` hook, next to `TracerProvider.shutdown()`.

**OpenInference ships no implementation.** The demo scripts carry a ~40-line local
store to prove the contract; real uploaders come from an application's own class, a
vendor SDK registering an entry point (the Arize SDK will), or — if upstream grows a
public byte-level uploader (§3.4) — a thin adapter over that. Implementations
targeting generic object stores should build on
[fsspec](https://filesystem-spec.readthedocs.io/) — the same transport layer
util-genai's hook uses — rather than per-store SDKs.

Configuration, in precedence order:

| mechanism | proposal |
|---|---|
| programmatic | `TraceConfig(blob_uploader=my_uploader)` — new optional field accepting any `BlobUploader` |
| entry point | group `openinference_blob_uploader`, selected by `OPENINFERENCE_BLOB_UPLOADER=<name>`; resolves an instance, class, or factory |

The entry-point mechanism, end to end: the package that owns the uploader registers
it in its packaging metadata (no import at install time) —

```toml
# e.g. arize_otel's pyproject.toml
[project.entry-points.openinference_blob_uploader]
arize = "arize_otel.blob:ArizeBlobUploader"
```

— and the operator sets `OPENINFERENCE_BLOB_UPLOADER=arize`. At `TraceConfig`
construction, if no uploader was passed in code, the name is resolved from the group,
instantiated if it is a class or zero-argument factory, and validated against the
protocol. Resolution failures log a warning and leave the uploader unset —
misconfiguration degrades to redaction, never crashes. The uploader's *own*
configuration (bucket, credentials) is its own concern, typically read from its own
env vars, keeping fully zero-code deployments possible.

### Two integration points, chosen per capture site

The door is picked per capture site by one question: *what form is the media in at
the moment the instrumentor is about to record it?* One instrumentor routinely uses
both (the OpenAI instrumentor does).

| Media at the capture site | Door |
|---|---|
| SDK hands you base64 (`image_url`, `input_audio.data`, `file_data`) | **point 1** — emit a data URI on the standard key; `mask()` does the rest |
| raw decoded bytes (realtime / websocket buffers) | **point 2** — `upload(Blob)` directly; skip the bytes → base64 → bytes round-trip |
| payload inside an already-serialized JSON attribute (`input.value`) | **point 2** — pre-pass before serialization |
| provider-hosted reference (`file.id`, `file_url`) | **neither** — record the reference verbatim; no bytes involved |

Whichever door, §2.4's policy applies identically, so a payload's fate depends only on
config, never on which door it entered through. **Only point 1, for images, ships in
#3409**; point 2 and the audio/file rows are the design for the deferred follow-ons.

### 2.2 Integration point 1 — the `TraceConfig` choke point (zero instrumentor changes)

`TraceConfig.mask()` already sees every `(key, value)` on `OITracer` spans and already
implements the >limit image redaction. The existing oversized-image branch upgrades
from redact-only to externalize-or-redact: an over-budget base64 image decodes to a
`Blob` and the uploader's URI is substituted in place, falling through to
`__REDACTED__` when there is no uploader or it declines. The image demo proves this
shape against *released* packages: the same attribute key carries `__REDACTED__` or a
storage URI depending only on the `TraceConfig` handed to the instrumentor. Any
instrumentor that uses `OITracer` — which the project requires — gets image offload
without a diff; audio/file keys join the same branch when their vocabulary lands.

**The `input.value` copy.** Instrumentors also JSON-serialize the whole raw request
into `input.value` *before* it reaches `mask()`, and `mask()` won't parse arbitrary
JSON to find base64 buried inside. The OpenAI instrumentor already redacts oversized
base64 images from that copy with an instrumentor-side pre-pass
(`redact_images_from_request_parameters`). Upgrading that pre-pass from redaction to
externalization through the same uploader — so the `input.value` copy carries the URI
too, with content addressing collapsing the double touch into one object — is open
question 1.

### 2.3 Integration point 2 — capture-time API for raw-bytes instrumentors (follow-on)

Not shipped in #3409 — this is the design for the audio follow-on. Instrumentors that
hold decoded bytes (openai-agents realtime `_finalize_user` / `_finalize_response`)
call `uploader.upload(Blob(data=wav_bytes, mime_type="audio/wav", attribute_key=...))`
directly when the encoded size would exceed budget, and record the returned URI (or
`__REDACTED__` on `None`) — never building a data URI at all. One migration behavior
change when it lands: over-budget audio that cannot upload becomes `__REDACTED__`
instead of today's truncated data URI — acceptable because the truncated payload was
cut mid-stream and never decodable. The audio demo applies this change to a live
realtime session by patching the released instrumentor.

### 2.4 Offload policy

One decision function, applied at both integration points, gates in strict order:
**hide → fits-inline-budget → upload → `__REDACTED__`**.

- **Privacy wins over upload.** Hidden content is never uploaded — uploading it would
  move PII into storage the operator explicitly asked to suppress. This intentionally
  diverges from gen_ai's "hook operates independently of capture opt-in flags": that
  clause exists so a hook can be the *sole* content sink in OTel's opt-in-capture
  world, whereas OI captures by default and its `hide_*` flags are redaction controls.
- **The inline budget is the existing one.** Images keep `base64_image_max_length`
  (back-compat); content that fits stays inline (tiny blobs aren't worth a fetch), and
  setting it to `0` offloads everything. The audio/file follow-on adds a
  `base64_media_max_length` budget and migrates realtime's private
  `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` onto it.
- **Over-budget content that cannot upload is `__REDACTED__`** — enabling an uploader
  only ever upgrades redaction to a URI.
- **Unsampled/non-recording spans skip upload** (`span.is_recording()` guard) — no
  paying for uploads nobody can see; a sampled-out span has no attribute to dangle.

### 2.5 Async model and failure modes

`upload()` stamps the URI now; bytes travel later (OTel's util-genai stamps refs the
same way).

| failure | behavior | who notices |
|---|---|---|
| storage unusable at construction | fail fast and loud: `ImportError` for a missing fsspec driver; recommended startup write probe → uploader disables itself, one log | operator; oversized media redacts, as with no uploader |
| bounded queue full (burst) | `upload()` returns `None` synchronously → attribute records `__REDACTED__` | span shows redacted content — never silently empty, never blocked |
| async write fails after URI stamped | retry-once then log; span carries a dangling URI | backend shows broken link for that one blob; transcript/mime survive on the span |
| process exits mid-queue | `atexit` → `shutdown(timeout_sec)` flush; still-pending blobs may be lost (dangling URIs) | same as above |
| uploader raises | caught at both integration points, logged, redaction fallback | never the application |
| memory pressure | bound = queue_capacity × largest blob; realtime turns are ~100s of KB → ~MBs at default capacity | operator tuning |

The dangling-URI window is the deliberate price of never blocking the hot path;
consumers should treat "object not (yet) there" as retryable. Deployments that cannot
tolerate it can pass a custom `BlobUploader` that writes synchronously.

### 2.6 Backend consumption (Phoenix)

URIs are ordinary string attributes — no Phoenix changes are required to store them.
Phoenix already renders URL-valued `message_content.image.image.url`; an audio player
and URI resolution (signed URLs, proxying, retention) are consumer follow-ons,
intentionally outside this design.

## 3. Compatibility with the OTel GenAI conventions

### 3.1 What the convention specifies

[`gen-ai-spans.md` § "Uploading content to external storage"](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md#uploading-content-to-external-storage)
(development status) says instrumentations **MAY** support in-process upload hooks and
leaves the hook API generic. Crucially, *"TODO: document a common approach to record
references to externally stored content"* is still open
([#45](https://github.com/open-telemetry/semantic-conventions-genai/issues/45)) —
**there is no normative reference-attribute convention to comply with.** An earlier
generic `BlobUploader` proposal
([python-contrib#3065](https://github.com/open-telemetry/opentelemetry-python-contrib/issues/3065):
`upload_async(blob) -> url`, the shape adopted here) was closed in favor of the
whole-payload `CompletionHook` after concerns about unsampled-span uploads and
in-memory buffering; §2.4's `is_recording()` guard and §2.5's bounded queue answer both.

### 3.2 Attribute mapping (audio, image, file)

The gen_ai message schemas define three binary content-part types, each carrying
`mime_type` and `modality` (`image|audio|video|document`): `blob` (inline base64),
`uri` (external reference — explicitly *"should not be a base64 data URL"*), and
`file` (provider-side pre-uploaded file id). OpenInference's dual-write already maps
OI URLs onto these — `data:` URL → `blob` part, anything else → `uri` part — so
**after blob upload the dual-write emits a `uri` part with no conversion changes.**
OI attribute keys don't change either; the URI replaces the data URI in place:

| content | OI attribute (key unchanged) | before | after | gen_ai dual-write |
|---|---|---|---|---|
| message image (**ships, #3409**) | `...message_content.image.image.url` | `data:image/png;base64,...` or `__REDACTED__` | storage URI | `blob` part → `uri` part (`modality:"image"`) |
| message audio (follow-on) | `...message_content.audio.audio.url` (+ `audio.mime_type`, `audio.transcript`) | `data:audio/wav;base64,...` | storage URI | `uri` part (`modality:"audio"`) via `_media_part_from_url` |
| message file / PDF (follow-on) | `...message_content.file.file.url` (+ `file.mime_type`, `file.name`); provider-hosted files as `file.id` | `data:application/pdf;base64,...` | storage URI | `uri` part (`modality:"document"`); `file.id` → `file` part |
| realtime audio (follow-on) | `input.audio.url` / `output.audio.url` + `.mime_type`, `.transcript` | truncated data URI | storage URI | n/a today (realtime spans are not dual-written) |

Only the image row ships now; the audio/file rows are the drafted mapping preserved on
`blob-upload-semconv` (#3450) for when their conventions land (open question 6).

Supporting details:

- **mime type:** audio and files carry `*.mime_type` fields; images embed mime in the
  data URI today, so uploaders MUST append a mime-derived extension (`….png`) to keep
  the URI self-describing.
- **size:** upstream is discussing `byte_size` on content parts
  ([PR #143](https://github.com/open-telemetry/semantic-conventions-genai/pull/143));
  adopt if it lands rather than inventing an OI field.
- **Caveat:** gen_ai defines `uri`/`file` parts as "what was sent to the provider",
  not "where telemetry offloaded the bytes". The convention explicitly allows hooks to
  rewrite message objects (blob → uri) and Google's deployed hook relies on it, but no
  convention blesses per-part rewrites yet — track
  [#45](https://github.com/open-telemetry/semantic-conventions-genai/issues/45) /
  [#304](https://github.com/open-telemetry/semantic-conventions-genai/issues/304) and
  re-align if they land (open question 3).
- **Conformance:** add a uri-part scenario to the Weaver `registry live-check` harness
  and schema-validate converted parts against the vendored GenAI JSON schemas.

### 3.3 Alternative considered: whole-payload refs

| | util-genai (whole payload) | Langfuse / Traceloop / **this proposal** (per part) |
|---|---|---|
| what uploads | entire messages JSON per invocation | only the binary part's bytes |
| span afterwards | content attrs *plus* `*_ref` attrs | same attribute keys, URI values |
| text queryability | moves to storage with everything else | text stays inline and queryable |
| backend rendering | needs new `_ref`-aware UI | Phoenix already renders URL-valued `image.image.url` |
| dedup | uuid per invocation | content hash dedups the same image resent every turn |

Per-part rewrite wins because OI's attribute model is already flat per-part URL fields
that accept both `data:` and external URIs, and Phoenix renders them today. A
whole-payload `messages_ref` equivalent is compatible follow-on work.

### 3.4 Code-reuse assessment: util-genai's uploader internals

Maintainer question: can OpenInference reuse
[util-genai's `_upload/completion_hook.py`](https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/_upload/completion_hook.py)
instead of shipping its own fsspec plumbing? **Resolution (post-review): OI ships no
plumbing at all — interface and policy only.** Their `UploadCompletionHook` cannot
carry per-part bytes, and OI should not maintain a fork of what it can't reuse: it has
no byte-level API (its `types.Blob` parts get base64-encoded *inside* the messages
JSON), writes text-mode JSON only, names by `uuid4()` rather than content hash, stamps
`*_ref` attributes on the span itself, and lives in a private module of an
experimental 0.x package.

What *is* shared — deliberately: the entry-point loading mechanics
(`openinference_blob_uploader` mirrors `opentelemetry_genai_completion_hook`), the
stamp-refs-before-bytes-land async model, the §2.5 contract behaviors their hook
exhibits (write probe, bounded drop-on-full queue, shutdown flush), and the
recommendation that object-store implementations build on fsspec.

Convergence path: if upstream exposes a *public* byte-level uploader (e.g. a
[python-contrib#3065](https://github.com/open-telemetry/opentelemetry-python-contrib/issues/3065)
revival), it slots behind the unchanged `BlobUploader` protocol, non-breaking. The
type-level groundwork exists:
[`opentelemetry.util.genai.types.Blob`](https://github.com/open-telemetry/opentelemetry-python-genai/blob/85fb8a6ce2c239a5009a94c71f39875cb84b7bee/util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py#L160-L171)
carries the same payload triple as ours (`content: bytes` ≙ `data`, `mime_type`,
`modality`), so an adapter is a one-liner in either direction. What upstream lacks is
not the type but any code that moves one `Blob`'s bytes to storage individually.

## 4. Demos

Two live scripts under [`scripts/`](./scripts/README.md) (`uv run --script …` against
a local Phoenix, `OPENAI_API_KEY` required), both driving real instrumented OpenAI
Agents SDK apps: `image_blob_demo.py` demonstrates the shipped scope — the same
vision-agent request run twice with only the `TraceConfig` changed (redaction vs.
storage URI on the same attribute key); `audio_blob_demo.py` illustrates the deferred
§2.3 capture-site design against a live Realtime API session (`--inline` shows today's
truncation). The demo blob store (`LocalBlobStore` in `common.py`) is a deliberate
mock — content-addressed local files.

## 5. JS / Java follow-on (not implemented)

The design ports directly — both ecosystems have the same two seams. JS
(`@arizeai/openinference-core`): the same `blobUploader` config field and mask
choke-point branch. Java (`TraceConfig`): same field + branch, `BlobUploader` as an
interface. Attribute keys and policy are language-independent.

## 6. Out of scope

- Production object stores (S3/GCS credentials, signed URL issuance, retention/GC,
  encryption at rest).
- Audio and file/PDF offload — deferred with their vocabulary (open question 6).
- Phoenix UI changes (audio player, blob-store proxy).
- Whole-payload `messages_ref`-style offload (§3.3).
- The openai-agents realtime migration itself (§2.3 is the design; landing it is a
  follow-on PR).

## 7. Open questions

1. **Images inside `input.value`.** The existing image pre-pass redacts; the
   generalized audio/file pre-pass externalizes. Unify by teaching the image pre-pass
   to externalize too, so `input.value` carries the URI instead of a redaction marker.
2. **Audio attribute keys.** `input.audio.*` / `output.audio.*` are audio attributes
   (today emitted only by the GPT realtime instrumentor). Promote to semconv as-is, or
   migrate their emitters onto `message_content.audio`.
3. **Upstream refs convention.** If
   [semantic-conventions-genai#45](https://github.com/open-telemetry/semantic-conventions-genai/issues/45)
   standardizes reference attributes (or #304 standardizes parts), revisit the §3.2
   mapping before GA.
4. **Should `hide_*` + uploader mean "archive but don't show"?** Deliberately answered
   "no" (privacy wins); a separate `archive_hidden_content` flag could add it later
   without breaking this design.
5. **Whole-payload text offload via util-genai's `CompletionHook`.** Oversized *text*
   (million-token contexts) is the same problem class and exactly what
   `UploadCompletionHook` was built for. OI's dual-write already materializes
   `gen_ai.input.messages` / `output.messages` at span end and could drive
   `load_completion_hook()` directly — their transport, their `*_ref` attributes, zero
   OI upload code. Prerequisites: gate behind OI's `hide_*` flags, and decide what
   searchable semantics remain on the span once text moves to storage.
6. **Audio/file message-content vocabulary and capture.** The deferred remainder of
   this design: `MESSAGE_CONTENT_AUDIO` / `MESSAGE_CONTENT_FILE` / `FileAttributes`
   semconv, their spec text and gen_ai dual-write mapping (drafted on
   `blob-upload-semconv`, [#3450](https://github.com/Arize-ai/openinference/pull/3450)),
   the `base64_media_max_length` budget + audio/file hide flags and mask rules, and the
   OpenAI audio/file capture layer (drafted on `blob-upload-openai`,
   [#3410](https://github.com/Arize-ai/openinference/pull/3410)). Blocked on settling
   the conventions: audio is emitted by one instrumentor under instrumentor-local keys
   (question 2), and files/PDF have no OI convention yet.
