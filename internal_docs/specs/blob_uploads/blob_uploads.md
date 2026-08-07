# Blob upload for large multimodal span content

**Status:** accepted 2026-07, **experimental** — the uploader contract and attribute
semantics may change while the feature matures. Ships as a media-agnostic
**`BlobUploader` contract** plus an **image offload policy** in
`openinference-instrumentation`. No packaged uploader implementation
([follow-up](#5-follow-up-a-packaged-general-purpose-uploader)) and no
semantic-convention changes.
**Demo:** [`scripts/`](./scripts/README.md) — a live image demo against a local Phoenix
(requires `OPENAI_API_KEY`), driving the shipped code via a `uv` path source.
**Scope:** Python. Images now; other media types via [the checklist](#6-checklist-adding-offload-support-for-a-new-media-type). JS/Java port
directly when needed — both have the same `TraceConfig`/mask seam.

## Overview

- **Problem:** multimodal content rides on spans as inline `data:<mime>;base64,...`
  values that size guards destroy (`__REDACTED__`, truncation) or that blow span
  sizes past OTLP limits.
- **Design:** a pluggable **`BlobUploader`** — decoded bytes go to external storage
  at capture time; the span attribute records only the destination URI. Same key,
  new value: `...message_content.image.image.url` = `"https://…/3a7bd3….png"`
  instead of `"__REDACTED__"`.
- **New public API** in `openinference-instrumentation` — interface and policy only,
  no transport:

  ```python
  Blob(data, mime_type, modality="", attribute_key=None, content_sha256="")  # frozen dataclass
  BlobUploader              # @runtime_checkable Protocol: upload(blob) -> Optional[str]; shutdown(timeout_sec)
  load_blob_uploader(name)  # resolves the "openinference_blob_uploader" entry-point group
  ```

- **One new `TraceConfig` field:** `blob_uploader` — set in code, or zero-code via
  `OPENINFERENCE_BLOB_UPLOADER=<name>` naming a registered entry point (code beats
  env). Existing knobs do the rest: `base64_image_max_length` says *what* offloads,
  `hide_input_images` what never leaves the process.
- **One policy,** inside the existing `TraceConfig.mask()` choke point:
  **hide → fits inline budget → upload → `__REDACTED__`** — no uploader, rejection,
  and failure all degrade to today's redaction, and never block the instrumented call.
- **Scope: images only; no semconv changes.** Image is the only media type with a
  settled, widely emitted convention. Audio (one instrumentor, instrumentor-local
  keys) and file/PDF (no convention yet) follow via [the checklist](#6-checklist-adding-offload-support-for-a-new-media-type); the `Blob`
  contract is media-agnostic, so future types add vocabulary and mask rules, not API.
- **gen_ai compatible:** the dual-write already maps non-`data:` URLs to `uri`
  parts, so an externalized image emits a spec-shaped part with zero conversion
  changes ([GenAI compatibility](#3-compatibility-with-the-otel-genai-conventions)).
- **No shipped uploader:** implementations come from applications or vendor SDKs; a
  packaged fsspec uploader is the expected [follow-up](#5-follow-up-a-packaged-general-purpose-uploader).

## 1. Problem

OpenInference spans capture multimodal content **inline** as `data:<mime>;base64,...`
attribute values. That worked for occasional small images; it does not survive contact
with production multimodal traffic. Realtime voice makes the arithmetic vivid: the
OpenAI Realtime API streams 24 kHz mono PCM16 both directions — 64 KB/s per side once
base64-encoded — so the 32,000-char inline cap preserves roughly half a second of
audio, cut mid-stream into undecodable base64. Vision traffic hits the same wall:
one real photo is a several-hundred-KB PNG. Production choices today:

| today's option | consequence |
|---|---|
| default redaction/truncation | content destroyed — a >32 KB image becomes `__REDACTED__` |
| raise the max-length env vars | a 663 KB PNG becomes an 884 KB attribute; multi-MB spans stress OTLP payload limits (gRPC default rejects the whole 4 MB+ batch), collectors, and span stores |
| hide flags (`OPENINFERENCE_HIDE_INPUT_IMAGES`, …) | attribute never emitted — no observability at all |

The missing option, and the subject of this spec: **upload the decoded bytes to
external storage at capture time and record only the destination URI on the span**.

Two structural facts shape the design:

- Every attribute set on an `OITracer`-created span already flows through a single
  choke point, `TraceConfig.mask(key, value)`, where image redaction happens today —
  so offload can happen there with **zero instrumentor changes**.
- Some capture sites (realtime PCM buffers) hold **raw bytes**, not data URIs.
  Forcing them through a data-URI round-trip just so the choke point can decode it
  again would double memory churn on a hot path — the contract must also support
  being called directly ([the capture-time door](#23-capture-time-door-for-raw-bytes-instrumentors-not-yet-used)).

## 2. Design

### 2.1 Interface

New module in `openinference-instrumentation`, public exports `Blob`, `BlobUploader`,
`load_blob_uploader`:

```python
@dataclass(frozen=True)
class Blob:
    data: bytes                          # decoded bytes — never base64 text
    mime_type: str                       # "image/png", "audio/wav", ...
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

Contract for implementations:

- **One `Blob` per media content part; a single uploader serves every mime type.**
  There is no per-mime registry: `blob.mime_type` / `blob.modality` let an
  implementation route (e.g. per-modality prefixes) or refuse (return `None` → that
  part alone redacts; siblings unaffected) as its own policy.
- **`upload` MUST return quickly.** Compute the destination URI synchronously
  (content-hash naming makes it computable before any I/O); transfer bytes on a
  background worker. Capture sites can sit on hot paths.
- **`None` means "not uploaded — redact".** On backpressure, after shutdown, or by
  policy, the caller records `__REDACTED__` — the same value oversized content gets
  with no uploader at all.
- **The returned reference MUST be a valid absolute URI** (a scheme is required) and
  SHOULD be the most consumer-resolvable form available — an `https://` or signed URL
  where possible; storage-scheme URIs (`gs://`, `s3://`) are valid canonical
  references but require viewer-side resolution. The caller validates the returned
  value and redacts invalid ones (bare file paths never reach span attributes).
  Consumers without access treat the value as they would `__REDACTED__`; teaching
  Phoenix/Arize to resolve storage-scheme URIs at render time is consumer-side
  follow-up work.
- **Errors never propagate.** The caller catches everything ([the `mask()` choke point](#22-the-mask-choke-point)), but
  implementations should also fail fast at construction when the destination is
  unusable, and log rather than raise from workers.
- **Content-hash naming is the recommended default**
  (`{base_path}/{sha256(data)}.{ext}`, extension mapped from the mime type):
  identical payloads dedup — significant for multi-turn chats that resend the same
  image every turn — and the URI is known before the bytes land.
- **`shutdown(timeout_sec)`** flushes pending work and stops workers. Core does not
  call it (flush ordering against the span exporter can't be guaranteed from there):
  implementations own their exit flush — an `atexit` hook or daemon-thread drain —
  following the `BatchSpanProcessor` precedent of the worker-owner owning shutdown.

**Configuration.** Programmatically, `TraceConfig(blob_uploader=my_uploader)` accepts
any object satisfying the protocol. Zero-code, the package that owns an uploader
registers it in its packaging metadata (no import at install time):

```toml
# e.g. a vendor SDK's pyproject.toml
[project.entry-points.openinference_blob_uploader]
arize = "arize_otel.blob:ArizeBlobUploader"
```

and the operator sets `OPENINFERENCE_BLOB_UPLOADER=arize`. At `TraceConfig`
construction, if no uploader was passed in code, the name is resolved from the
entry-point group, instantiated if it is a class or zero-argument factory, and
validated against the protocol. **Loads are memoized per name**, so the N
instrumentors in a process (each constructing its own default `TraceConfig`) share
one uploader instance — one worker pool, one queue — rather than N. Resolution
failures (unknown name, wrong type, import error) log a warning and leave the
uploader unset — env misconfiguration degrades to redaction, never crashes. A
wrong-typed `blob_uploader` passed *in code* (a string, a class, an object missing
the methods) raises `TypeError` at construction — a programmer error fails fast
instead of being silently swallowed at mask time. The uploader's *own* configuration
(bucket, credentials) is its own concern, typically read from its own env vars,
keeping fully zero-code deployments possible.

### 2.2 The `mask()` choke point

`TraceConfig.mask()` already sees every `(key, value)` set on `OITracer` spans and
already implements the over-limit image redaction. The shipped change upgrades that
branch from redact-only to externalize-or-redact: an input- or output-message
`message_content.image.image.url` value that is a base64 data URI over
`base64_image_max_length` is decoded to a `Blob` and replaced by the uploader's URI,
falling through to `__REDACTED__` when there is no uploader, the uploader declines,
the returned reference is not a valid URI, or anything raises. Lazy (callable)
attribute values are resolved before the size check, so an oversized image cannot
bypass the budget by arriving as a callable. Any instrumentor that uses `OITracer` —
which the project requires — gets image offload without a diff.

Externalization runs **exactly once per attribute, and only for spans that will be
seen**. `OITracer.start_span` masks attributes for the *sampler* with a pure,
side-effect-free view (oversized images appear as `__REDACTED__` to sampling); the
uploader-enabled mask runs once in `OpenInferenceSpan.set_attribute` after the span
exists, which skips masking side effects entirely for suppressed and sampled-out
(non-recording) spans. Post-end attribute writes still surface the SDK's
"setting attribute on ended span" warning.

Policy properties, in gate order:

- **Privacy wins over upload.** The `hide_*` branches run first — hidden content is
  never uploaded, because uploading it would move PII into storage the operator
  explicitly asked to suppress. This intentionally diverges from the OTel GenAI
  guidance that upload hooks run independently of capture opt-ins ([convention details](#31-what-the-convention-specifies)): that clause
  exists so a hook can be the *sole* content sink in OTel's opt-in-capture world,
  whereas OpenInference captures by default and its `hide_*` flags are redaction
  controls.
- **The inline budget is the existing one.** Content within
  `base64_image_max_length` stays a data URI (tiny blobs aren't worth a fetch);
  setting the budget to `0` offloads everything.
- **Enabling an uploader only ever upgrades redaction to a URI** — over-budget
  content that cannot upload is `__REDACTED__`, exactly as before.
- **Non-recording (sampled-out) spans skip masking side effects entirely**
  (`OpenInferenceSpan.set_attribute` returns early) — no uploads nobody can see, and
  no dangling URI for a span that was never exported.

Only the `*.url` leaf is ever externalized — sibling attributes (mime type, names,
transcripts) never match the branch.

### 2.3 Capture-time door for raw-bytes instrumentors (not yet used)

Instrumentors that hold decoded bytes (e.g. realtime PCM buffers) skip the data-URI
round-trip and call `uploader.upload(Blob(data=..., mime_type=..., attribute_key=...))`
directly when the encoded size would exceed budget, recording the returned URI or
`__REDACTED__`. The same policy order applies at that door. No shipped instrumentor
uses this yet; it lands with audio support ([checklist](#6-checklist-adding-offload-support-for-a-new-media-type)).

### 2.4 Async model and failure semantics

`upload()` stamps the URI now; bytes travel later (OTel util-genai's hook stamps refs
the same way — [convention details](#31-what-the-convention-specifies)). What the caller guarantees versus what implementations must
handle:

| failure | behavior | who notices |
|---|---|---|
| uploader raises or value undecodable | caught in `mask()`, logged, `__REDACTED__` | never the application |
| uploader refuses (queue full, shutdown, policy) | `upload()` returns `None` synchronously → `__REDACTED__` | span shows redacted content — never silently empty, never blocked |
| async write fails after URI stamped | implementation logs; span carries a dangling URI | backend shows a broken link for that one blob |
| process exits mid-queue | implementation-owned `atexit` flush (core never calls `shutdown()`); still-pending blobs may be lost | same as above |
| memory pressure | bound = implementation queue capacity × largest blob | operator tuning |

The dangling-URI window is the deliberate price of never blocking the hot path;
consumers should treat "object not (yet) there" as retryable. Deployments that cannot
tolerate it can pass an uploader that writes synchronously.

**Backend consumption:** URIs are ordinary string attributes — ingestion is
unchanged, and Phoenix already renders `http(s)`-valued `image.image.url`. Decision
on display semantics: the span carries the **most resolvable URI the uploader can
produce** (signed/`https` where possible); resolving canonical storage-scheme URIs
(`gs://` → authenticated fetch) at render time is consumer-side follow-up work in
Phoenix/Arize, not an attribute-level concern.

## 3. Compatibility with the OTel GenAI conventions

### 3.1 What the convention specifies

[`gen-ai-spans.md` section "Uploading content to external storage"](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md#uploading-content-to-external-storage)
(development status) says instrumentations **MAY** support in-process upload hooks and
leaves the hook API generic. Crucially, *"TODO: document a common approach to record
references to externally stored content"* is still open
([#45](https://github.com/open-telemetry/semantic-conventions-genai/issues/45)) —
**there is no normative reference-attribute convention to comply with.** An earlier
generic `BlobUploader` proposal
([python-contrib#3065](https://github.com/open-telemetry/opentelemetry-python-contrib/issues/3065):
`upload_async(blob) -> url`, the shape adopted here) was closed in favor of the
whole-payload `CompletionHook` after concerns about unsampled-span uploads and
in-memory buffering; [the non-recording guard](#22-the-mask-choke-point) and the bounded-queue
contract in [the async model](#24-async-model-and-failure-semantics) answer both.

### 3.2 Message-part mapping

The gen_ai message schemas define three binary content-part types, each carrying
`mime_type` and `modality` (`image|audio|video|document`): `blob` (inline base64),
`uri` (external reference — explicitly *"should not be a base64 data URL"*), and
`file` (provider-side pre-uploaded file id). OpenInference's dual-write already maps
OI URLs onto these — `data:` URL → `blob` part, anything else → `uri` part — so
**after blob upload the dual-write emits a `uri` part with no conversion changes**:
`{"type": "uri", "modality": "image", "uri": "s3://…"}`. Future media types map the
same way ([checklist](#6-checklist-adding-offload-support-for-a-new-media-type)); provider-hosted file ids map to `file` parts,
references with no bytes.

Two upstream details tracked deliberately: gen_ai defines `uri`/`file` parts as "what
was sent to the provider", not "where telemetry offloaded the bytes" — the convention
allows hooks to rewrite message objects (blob → uri) and Google's deployed hook relies
on that mutability, but no convention blesses per-part rewrites yet (re-align if
[#45](https://github.com/open-telemetry/semantic-conventions-genai/issues/45) /
[#304](https://github.com/open-telemetry/semantic-conventions-genai/issues/304)
land). And upstream is discussing `byte_size` on content parts
([semconv-genai PR #143](https://github.com/open-telemetry/semantic-conventions-genai/pull/143));
adopt that rather than inventing an OI field.

### 3.3 Alternative considered: whole-payload refs

| | util-genai (whole payload) | Langfuse / Traceloop / **this design** (per part) |
|---|---|---|
| what uploads | entire messages JSON per invocation | only the binary part's bytes |
| span afterwards | content attrs *plus* `*_ref` attrs | same attribute keys, URI values |
| text queryability | moves to storage with everything else | text stays inline and queryable |
| backend rendering | needs new `_ref`-aware UI | Phoenix already renders URL-valued `image.image.url` |
| dedup | uuid per invocation | content hash dedups the same image resent every turn |

Per-part rewrite wins because OI's attribute model is already flat per-part URL fields
that accept both `data:` and external URIs, and Phoenix renders them today. The two
approaches also compose: whole-payload refs remain the right tool for oversized
*text* — driving util-genai's `CompletionHook` from OI's dual-write is possible
follow-up work, orthogonal to this design.

### 3.4 Why not reuse util-genai's uploader internals

The natural question: can OpenInference reuse
[util-genai's `_upload/completion_hook.py`](https://github.com/open-telemetry/opentelemetry-python-genai/blob/main/util/opentelemetry-util-genai/src/opentelemetry/util/genai/_upload/completion_hook.py)
instead of writing byte-upload plumbing? No — their `UploadCompletionHook` cannot
carry per-part bytes: it has no byte-level API (its `types.Blob` parts get
base64-encoded *inside* the messages JSON), writes text-mode JSON only, names by
`uuid4()` rather than content hash, stamps `*_ref` attributes on the span itself, and
lives in a private module of an experimental 0.x package. Rather than maintain a fork,
OpenInference ships the interface and policy only.

What *is* shared — deliberately: the entry-point loading mechanics
(`openinference_blob_uploader` mirrors `opentelemetry_genai_completion_hook`), the
stamp-refs-before-bytes-land async model, and the contract behaviors their hook
exhibits (write probe, bounded drop-on-full queue, shutdown flush). The type-level
groundwork for convergence also exists:
[`opentelemetry.util.genai.types.Blob`](https://github.com/open-telemetry/opentelemetry-python-genai/blob/85fb8a6ce2c239a5009a94c71f39875cb84b7bee/util/opentelemetry-util-genai/src/opentelemetry/util/genai/types.py#L160-L171)
carries the same payload triple as ours (`content: bytes` ≙ `data`, `mime_type`,
`modality`), so an adapter is a one-liner in either direction. What upstream lacks is
not the type but any code that moves one `Blob`'s bytes to storage individually.

## 4. Demo

Two live scripts under [`scripts/`](./scripts/README.md) (`uv run --script …`
against a local Phoenix, `OPENAI_API_KEY` required) resolve
`openinference-instrumentation` from **this repo** via a `[tool.uv.sources]` path in
their PEP 723 metadata, so they exercise the shipped
`Blob`/`BlobUploader`/`TraceConfig` code rather than a vendored copy — the same
chat-completions vision request run twice with only the `TraceConfig` changed
(redaction vs. storage URL on the same attribute key). Both uploaders explicitly
subclass `BlobUploader` and register their own `atexit` flush.
[`image_blob_demo.py`](./scripts/image_blob_demo.py)'s `LocalBlobStore` writes
content-addressed local files and serves them over a local HTTP port, so the
recorded reference is a real `http://localhost:…` URL Phoenix renders while the
script keeps serving. [`gcs_blob_demo.py`](./scripts/gcs_blob_demo.py) uploads to a
real Google Cloud Storage bucket (application-default credentials); an editable
`BASE64_IMAGE_MAX_LENGTH` constant reaches all three outcomes — inline, redact,
offload — and the recorded reference is a V4 signed URL when the credentials can
sign (service-account key, or `GCS_SIGNING_SERVICE_ACCOUNT` impersonation via IAM
signBlob), else the `https://storage.googleapis.com/…` form.

## 5. Follow-up: a packaged general-purpose uploader

Shipping a real uploader is deliberately deferred, but it is the expected next step
once the contract proves out. The natural candidate is an **fsspec-based byte
uploader**, directly analogous to util-genai's fsspec `UploadCompletionHook` (which
ships inside the util package behind an `[upload]` extra — precedent that an
interface package can carry a reference transport):

- writes via [fsspec](https://filesystem-spec.readthedocs.io/) so one implementation
  covers `s3://`, `gs://`, `file://`, and local paths with zero per-store code;
- content-addressed `{base_path}/{sha256}.{ext}` naming with per-mime `content_type`
  on writes (a `.wav`/`.png` object must be servable);
- bounded queue + background worker, drop-to-`None` on backpressure, a startup write
  probe that disables the uploader loudly if the destination is unusable, a bounded
  dedup cache, and `atexit`-wired `shutdown()`;
- packaged behind an optional extra (e.g.
  `openinference-instrumentation[blob-upload]`) and self-registered as an entry point
  (`fsspec = …`) whose zero-arg factory reads its base path from an env var —
  making the whole feature usable with two env vars and no code.

Alternatively, if upstream revives a public byte-level uploader
([python-contrib#3065](https://github.com/open-telemetry/opentelemetry-python-contrib/issues/3065)),
it slots behind the unchanged `BlobUploader` protocol as the recommended
implementation — non-breaking either way.

## 6. Checklist: adding offload support for a new media type

The shipped machinery is media-agnostic; what's missing per media type is vocabulary
and gates. To add offload for a new type (audio and file/PDF are the known next two),
work through:

1. **Semantic conventions.** Add the message-content constant
   (`MessageContentAttributes.MESSAGE_CONTENT_<TYPE>`) and a nested attribute class
   (`<Type>Attributes` with at minimum `<type>.url` and `<type>.mime_type`) in
   `openinference-semantic-conventions` (mirror in the Go package), plus rows and
   examples in `spec/semantic_conventions.md` and `spec/multimodal_attributes.md`.
   Nuances seen in drafting: audio already has `AudioAttributes`
   (`audio.url/mime_type/transcript`) but is emitted by one instrumentor under
   instrumentor-local `input.audio.*`/`output.audio.*` keys — promote or migrate
   those first; files need `file.name` and a `file.id` for provider-hosted
   references, which carry **no bytes** and must never enter the externalize path.
2. **`TraceConfig` gates.** Hide flags (`hide_input_<type>`, `hide_output_<type>`
   where outputs exist) and an inline budget — either reuse
   `base64_image_max_length`'s pattern or introduce a shared
   `base64_media_max_length` for non-image types — each with an
   `OPENINFERENCE_*` env var, documented in `spec/configuration.md`.
3. **`mask()` wiring.** Add the hide branches **before** the externalize branch
   (privacy wins by branch order), then extend the size-gated branch: match the
   message-content prefix and `endswith(<Type>Attributes.<TYPE>_URL)` — only the URL
   leaf externalizes — and route through the existing `_externalize_or_redact`.
   A generic `data:` matcher (`is_base64_media_url`) is needed once non-image mimes
   participate; `is_base64_url` only matches `data:image/`.
4. **gen_ai dual-write.** Map the content type in `_genai_conversion.py` (generalize
   `_image_part_from_url` into a modality-parameterized helper): data URI → `blob`
   part, external URL → `uri` part, provider file id → `file` part; add the modality
   value if missing (`document` for files). Schema-validate against the vendored
   GenAI JSON schemas and add a uri-part scenario to the Weaver `registry live-check`
   conformance harness.
5. **Instrumentor capture.** Emit the standard attribute shapes (door 1 — `mask()`
   then applies with zero further changes); raw-bytes capture sites call the uploader
   directly (door 2 — [the capture-time door](#23-capture-time-door-for-raw-bytes-instrumentors-not-yet-used)); and add an `input.value` pre-pass so the serialized
   request copy doesn't leak what the structured attributes mask (see open
   question 1).
6. **Tests.** Trace-config masking is a required test category: hide-beats-upload,
   within-budget stays inline, redact-without-uploader, redact-on-rejection,
   external URLs untouched, non-recording spans skip upload.

Drafts of steps 1–4 for audio and file exist in this repo's branch history and can be
revived when their conventions settle.
