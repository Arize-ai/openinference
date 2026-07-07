# Tech Spec: Large Multimodal Content Capture via Blob Upload

**Status:** Experimental (branch `experimental/multimodal-blob-upload`)
**Scope:** Python (`openinference-instrumentation`, `openinference-semantic-conventions`, `openinference-instrumentation-openai`), `spec/`
**Out of scope (this iteration):** JS/Java parity, Phoenix/backend rendering, URI resolution/proxying/signing

Layout of this spec:

```
multimodal_blob_upload/
├── techspec.md                       — this document: problem, design, API, decisions
└── scripts/
    ├── README.md                     — how to run the demo, what to look for
    └── openai_image_blob_upload.py   — custom-uploader demo, live API + Phoenix (uv-runnable)
```

## 1. Problem

Modern LLM APIs accept and produce large binary content — audio (OpenAI chat completions
`input_audio` / audio-modality responses), PDF documents (chat completions `file` parts,
Responses API `input_file`), and images. Instrumentation that captures this content inline
records multi-megabyte base64 strings in span attributes, which:

- exceeds OTLP transport limits (gRPC default is 4 MB per message — the whole span batch
  is rejected, not just the attribute);
- inflates backend storage (span attributes are stored verbatim);
- is destroyed by the existing size guard: values over
  `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` are replaced with `__REDACTED__`, so the
  content is unrecoverable.

Separately, before this branch the OpenAI instrumentor simply did not capture audio or
file content at all (`input_audio`, `input_file`, audio responses were TODOs), so there
was no observability for audio/document workloads.

## 2. Design overview

Two concerns, handled separately (mirroring the OTel GenAI semconv model):

1. **Capture normalization (per instrumentor).** Every extractor that meets binary
   content emits it in one canonical inline form: a `data:<mime>;base64,<payload>` URI on
   a media URL attribute (`…message_content.image.image.url`,
   `…message_content.audio.audio.url`, `…message_content.file.file.url`). Provider
   file ids (OpenAI Files API) are captured as `file.id` — a reference, never bytes.

2. **Externalization (central, config-driven).** A pluggable `BlobUploader` hook on
   `TraceConfig`, evaluated inside the existing `mask()` choke point that every attribute
   already passes through (`OpenInferenceSpan.set_attribute` → `TraceConfig.mask`). When
   an inline base64 data URI exceeds the size threshold and an uploader is configured,
   the decoded bytes are uploaded to external storage and the attribute records the
   destination URI instead. With no uploader, behavior degrades to today's redaction.
   Instrumentors never talk to storage.

This aligns exactly with the OTel GenAI semconv message-part model
(`gen_ai.input.messages` / `gen_ai.output.messages`, status Development), defined in the
dedicated [semantic-conventions-genai][genai-repo] repo — prose in
[`docs/gen-ai/gen-ai-spans.md`][genai-spans], normative part schemas in
[`model/gen-ai/gen-ai-input-messages.json`][genai-input-schema] /
[`gen-ai-output-messages.json`][genai-output-schema] (the `BlobPart`/`UriPart`/`FilePart`
types were added via [semantic-conventions#2754][semconv-pr-2754], closing
[semantic-conventions#1556][semconv-issue-1556]):

| OpenInference representation | GenAI message part |
|---|---|
| `data:<mime>;base64,…` in `*.url` | `{"type": "blob", "modality", "mime_type", "content"}` |
| any other URI in `*.url` (https, s3, gs, …) | `{"type": "uri", "modality", "mime_type"?, "uri"}` |
| `file.id` | `{"type": "file", "modality", "file_id"}` |

The GenAI spec ([`gen-ai-spans.md`][genai-spans]) explicitly recommends "store content
externally and record references on the spans" for production;
[`opentelemetry-util-genai`][util-genai-pypi] ships an experimental fsspec
[`UploadCompletionHook`][upload-hook-src] for whole-payload upload (in the
[opentelemetry-python-genai][python-genai-repo] repo). This design is the per-part
analog, and after externalization the existing dual-write conversion
(`enable_genai_semconv`) emits spec-conformant `uri` parts with no extra work.

## 3. Semantic convention additions (`spec/`, `openinference-semantic-conventions`)

- New message content type `"file"` alongside `text`/`image`/`audio`/`reasoning`/`tool_use`.
- New constants:
  - `MessageContentAttributes.MESSAGE_CONTENT_AUDIO = "message_content.audio"`
  - `MessageContentAttributes.MESSAGE_CONTENT_FILE = "message_content.file"`
  - `FileAttributes.FILE_URL = "file.url"` — http(s) URL, base64 data URI, or external
    storage URI
  - `FileAttributes.FILE_MIME_TYPE = "file.mime_type"`
  - `FileAttributes.FILE_NAME = "file.name"`
  - `FileAttributes.FILE_ID = "file.id"` — provider-assigned pre-uploaded file id
- `audio.url` / `file.url` explicitly admit three forms: http(s) URL, base64 data URI,
  external-storage URI. Spec text: `spec/multimodal_attributes.md`
  ("External Storage for Large Media"), `spec/semantic_conventions.md`,
  `spec/configuration.md`.

## 4. Core API (`openinference-instrumentation`)

New module `openinference/instrumentation/_blob_upload.py`; public exports `Blob`,
`BlobUploader`, `FsspecBlobUploader` from `openinference.instrumentation`.

```python
@dataclass(frozen=True)
class Blob:
    data: bytes                    # decoded payload
    mime_type: str                 # "audio/wav", "application/pdf", ...
    modality: str = ""             # "image"|"audio"|"video"|"document"; derived from mime
    attribute_key: Optional[str] = None   # span attribute the content came from
    content_sha256: str = ""       # hex digest of data; computed automatically

@runtime_checkable
class BlobUploader(Protocol):
    def upload(self, blob: Blob) -> Optional[str]:
        """Return the destination URI immediately; write asynchronously.
        Return None if the blob cannot be accepted (caller falls back to
        redaction). Must never block the instrumented call path."""
    def shutdown(self, timeout_sec: float = 10.0) -> None: ...

class FsspecBlobUploader:  # implements BlobUploader
    def __init__(self, base_path: str, *, max_queue_size: int = 20): ...
```

### FsspecBlobUploader in depth

**What fsspec is and why it was chosen.** [fsspec](https://filesystem-spec.readthedocs.io/)
("filesystem spec") is the de-facto Python abstraction for file-like storage: one
`AbstractFileSystem` API implemented by pluggable backends for local disk, S3, GCS,
Azure, HTTP, in-memory, and dozens more. It is the same storage layer pandas, dask, and
Hugging Face datasets use to accept `s3://…` paths anywhere a filename is expected — and,
importantly for convergence, it is what OTel's experimental
[`UploadCompletionHook`][upload-hook-src] in `opentelemetry-util-genai` builds on
(`fsspec.url_to_fs(base_path)`). Building the default uploader on fsspec means:

- one implementation covers every deployment target — the user changes a URI string,
  not code;
- credentials/config ride on each backend's standard mechanisms (e.g. `s3fs` honors
  `AWS_*` env vars / `~/.aws`, `gcsfs` honors Application Default Credentials) — the
  uploader itself never touches secrets;
- the storage dependency stays optional and user-chosen: the core package does not pull
  in boto3/google-cloud-storage; the user installs only the driver for their scheme.

| `base_path` example | Backend package needed | Notes |
|---|---|---|
| `/var/oi-media`, `file:///var/oi-media` | none | works even without fsspec installed (plain-`pathlib` fallback) |
| `s3://bucket/prefix` | `s3fs` | credentials via standard AWS config chain |
| `gs://bucket/prefix` | `gcsfs` | credentials via Application Default Credentials |
| `abfs://container/prefix` | `adlfs` | Azure Blob / Data Lake |
| `memory://oi-media` | none (ships with fsspec) | useful in tests |

fsspec itself is the `blob-upload` extra (`pip install
openinference-instrumentation[blob-upload]`); at construction time
`fsspec.url_to_fs(base_path)` resolves the URI scheme to a concrete filesystem object
plus a root path. If fsspec is missing, local/`file://` destinations silently fall back
to direct `pathlib` writes, while remote schemes raise an `ImportError` naming the extra
— failing fast at setup rather than dropping blobs at runtime.

**Upload path, step by step.** `upload(blob)` runs on the instrumented call path, so
everything it does is O(hash) and memory-bounded:

1. Compute the destination name from the content: `{sha256(data)}{ext}`, extension
   mapped from the mime type (`audio/wav → .wav`, `application/pdf → .pdf`, …,
   `.bin` fallback). The full URI `{base_path}/{name}` is therefore **deterministic
   before any I/O happens** — this is what lets `upload()` return synchronously while
   the bytes travel later.
2. Dedup check: if this digest was already enqueued, return the same URI immediately
   (identical prompts re-sent across spans upload once).
3. Enqueue `(path, blob)` on a bounded queue (`max_queue_size`, default 20, clamped
   to ≥1 because `Queue(0)` would mean *unbounded* in Python). If the queue is full —
   storage slower than the app is producing media — `upload()` returns `None` and the
   caller records `__REDACTED__` instead: bounded memory beats unbounded buffering of
   multi-megabyte payloads.
4. A single daemon worker thread drains the queue and writes via
   `filesystem.pipe_file(path, data)` (an atomic whole-object write), creating missing
   parent directories first for directory-like backends (object stores have no real
   directories; a failed `makedirs` there is ignored and the write itself is
   authoritative).
5. `shutdown(timeout_sec)` flushes pending writes and stops the worker; afterwards
   `upload()` returns `None` (→ redaction). Applications should call it at exit, e.g.
   next to `TracerProvider.shutdown()`.

**Failure semantics.** The guiding rule is that telemetry must never break or block the
instrumented application:

| Failure | Behavior |
|---|---|
| queue full at capture time | `upload()` → `None`; attribute records `__REDACTED__` |
| backend write fails (network, permissions) | logged; the span keeps the already-recorded URI, which now dangles — bounded by `max_queue_size` in-flight blobs |
| process dies before the queue drains | same as above: at most `max_queue_size` dangling URIs |
| remote scheme without fsspec/driver installed | `ImportError` at uploader construction (fail-fast) |

The dangling-URI window is the deliberate price of never blocking the hot path; a
consumer resolving a URI should treat "object not (yet) there" as retryable. Deployments
that cannot tolerate it can pass a custom `BlobUploader` that writes synchronously
(see the demo script's `MyBlobUploader`).

### TraceConfig integration

New fields / env vars (precedence: code > env > default, as with all existing fields):

| Field | Env var | Default |
|---|---|---|
| `hide_input_audio` | `OPENINFERENCE_HIDE_INPUT_AUDIO` | `False` |
| `hide_output_audio` | `OPENINFERENCE_HIDE_OUTPUT_AUDIO` | `False` |
| `hide_input_files` | `OPENINFERENCE_HIDE_INPUT_FILES` | `False` |
| `base64_media_max_length` | `OPENINFERENCE_BASE64_MEDIA_MAX_LENGTH` | `32_000` |
| `blob_uploader` | `OPENINFERENCE_BLOB_UPLOAD_BASE_PATH` (+ `OPENINFERENCE_BLOB_UPLOAD_MAX_QUEUE_SIZE`) | `None` |

`base64_image_max_length` is unchanged for images (back-compat); the new
`base64_media_max_length` governs audio/file/video data URIs. Setting
`OPENINFERENCE_BLOB_UPLOAD_BASE_PATH` constructs a default `FsspecBlobUploader`; passing
`blob_uploader=` in code takes precedence and accepts any `BlobUploader` implementation.

The env var names deliberately parallel OTel's experimental
`OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH` / `..._UPLOAD_MAX_QUEUE_SIZE`
(defined by [`opentelemetry-util-genai`][util-genai-pypi]'s upload hook,
[source][upload-hook-src]) so a future aliasing is mechanical.

### mask() decision table

Evaluated per attribute in `OpenInferenceSpan.set_attribute`, in order:

```
hide_input_audio  + input  message audio key            → drop (None)
hide_output_audio + output message audio key            → drop (None)
hide_input_files  + input  message file key             → drop (None)
image data URI > base64_image_max_length on image.url   → upload → URI, else __REDACTED__
media data URI > base64_media_max_length on audio.url
  or file.url (input or output messages)                → upload → URI, else __REDACTED__
otherwise                                               → unchanged
```

Invariants:
- Hide settings take precedence over externalization — hidden content is never uploaded.
- Small payloads (≤ threshold) stay inline as data URIs (GenAI `blob` parts).
- Upload failure/rejection degrades to `__REDACTED__` (today's behavior), never blocks.

## 5. OpenAI instrumentor capture (`openinference-instrumentation-openai`)

New helpers `_media.py` (format→mime maps, data-URI builders) and `_media_utils.py`
(request-payload processing). Capture added:

| Surface | Content | Attributes emitted |
|---|---|---|
| Chat Completions request | `input_audio` part | `message_content.type="audio"`, `audio.url` (data URI), `audio.mime_type` |
| Chat Completions request | `file` part | `message_content.type="file"`, `file.url` (data URI, from `file_data`), `file.mime_type` (guessed from `filename`), `file.name`, `file.id` |
| Chat Completions response | `message.audio` (`ChatCompletionAudio`) | `message.contents.0.…type="audio"`, `audio.url` (data URI; format read from request `audio.format`), `audio.mime_type`, `audio.transcript`, `message_content.id` |
| Responses API request | `input_audio` part | as chat `input_audio` |
| Responses API request | `input_file` part | `file.url` (from `file_url`, else data URI from `file_data`), `file.mime_type`, `file.name`, `file.id` |

Second path — `input.value`: the raw request params are also JSON-serialized into
`input.value`. `redact_media_from_request_parameters` (analogous to the existing image
redaction) walks chat `messages[].content[]` and Responses `input[].content[]`, and for
`input_audio.data` / `file.file_data` / `input_file.file_data` payloads applies the same
hide → externalize → redact policy before serialization. With an uploader configured the
payload field inside `input.value` carries the destination URI. Content addressing makes
the double touch (structured attribute + `input.value`) resolve to one upload and one URI.

## 6. GenAI dual-write conversion (`_genai_conversion.py`)

`_image_part_from_url` generalized to `_media_part_from_url(url, modality, mime_type=None)`;
new `_file_part_from_id`. Conversion coverage (both the flattened-attribute path and the
marshaled-JSON path):

- `message_content.audio.audio.url` → `blob` (data URI) or `uri` part, `modality="audio"`,
  carrying `audio.mime_type` when present.
- `message_content.file.file.url` → `blob`/`uri` part, `modality="document"`, carrying
  `file.mime_type`.
- `message_content.file.file.id` (no url) → `file` part with `file_id`.
- `GenAIModalityValues` gains `DOCUMENT = "document"` (matches the semconv
  [`Modality` enum][genai-input-schema]: `image | video | audio | document`).

Output is validated in tests against the vendored OTel GenAI JSON schemas
(`tests/fixtures/genai_schemas/gen-ai-{input,output}-messages.json`, vendored from
[semantic-conventions v1.41.1][semconv-v1411]), which already define
`BlobPart`/`UriPart`/`FilePart`.

## 7. End-to-end example

A short runnable demo of the flow below — a custom `BlobUploader` on a live vision
call, exported to Phoenix — lives in
[`scripts/openai_image_blob_upload.py`](./scripts/openai_image_blob_upload.py)
(`uv run --script …`; see [`scripts/README.md`](./scripts/README.md)).

```python
from openinference.instrumentation import TraceConfig, FsspecBlobUploader
from openinference.instrumentation.openai import OpenAIInstrumentor

config = TraceConfig(
    blob_uploader=FsspecBlobUploader(base_path="s3://my-bucket/oi-media"),
)
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=config)
```

A chat completion with a 5 MB WAV input produces (audio over the 32k default threshold):

```
llm.input_messages.0.message.contents.1.message_content.type       = "audio"
llm.input_messages.0.message.contents.1.…audio.audio.url           = "s3://my-bucket/oi-media/3a7bd3….wav"
llm.input_messages.0.message.contents.1.…audio.audio.mime_type     = "audio/wav"
```

and, with `enable_genai_semconv=True`, `gen_ai.input.messages` contains
`{"type": "uri", "modality": "audio", "mime_type": "audio/wav", "uri": "s3://…/3a7bd3….wav"}`.

Without an uploader the same span carries `__REDACTED__` (unchanged from today, except
audio/files are now captured at all). Under the threshold, the data URI stays inline.

## 8. Testing

- `openinference-instrumentation/tests/test_blob_upload.py` — uploader semantics
  (content addressing, dedup, shutdown, protocol conformance), all mask() paths
  (upload/redact/inline/hide/precedence/env-var construction), custom uploader injection.
- `openinference-instrumentation/tests/test_genai.py::test_get_genai_attributes_maps_audio_and_file_parts`
  — blob/uri/file part conversion, schema-validated.
- `openinference-instrumentation-openai/tests/.../test_multimodal_media.py` — extractor
  coverage for all five capture surfaces plus `input.value` redaction/externalization.

Known pre-existing (unrelated) issue: `test_tool_calls.py::test_tool_calls` fails when the
openai suite runs in one process on this branch's base commit as well (span-count
pollution from test ordering).

## 9. Design decisions & alternatives considered

- **Per-part URI substitution over whole-payload `*_ref` upload.** OTel's
  [`UploadCompletionHook`][upload-hook-src] uploads the entire inputs/outputs JSON and
  stamps `gen_ai.input.messages_ref` (implementation-defined names, not in the semconv
  registry — a generic `*.blob_ref.*` convention was proposed in
  [semantic-conventions#1521][semconv-pr-1521] but closed unmerged). Per-part
  substitution keeps text/tool-call content queryable inline, works with OpenInference's
  flattened attribute model, and produces registry-shaped `uri` parts. A whole-payload
  `input.value.ref` escape hatch remains a possible follow-up.
- **Synchronous URI, asynchronous write.** Content addressing (SHA-256) makes the
  destination deterministic, so the attribute can be set without waiting for storage.
  Consequence: a URI may briefly dangle if the process dies before the queue drains
  (bounded by `max_queue_size`); acceptable for telemetry.
- **Protocol, not base class.** Any object with `upload`/`shutdown` works
  (`@runtime_checkable`), enabling adapters over OTel's BlobUploader work
  (Google's proposal in [opentelemetry-python-contrib#3065][contrib-3065]:
  `upload_async(blob) -> uri` returning the destination immediately with a background
  write — the same shape adopted here), OpenLLMetry-style image uploaders, or in-house
  stores without inheriting from us.
- **Hide-wins ordering.** Privacy settings must not be weakened by externalization:
  content the user asked to hide never leaves the process.
- **`is_base64_url` untouched.** The image-only helper keeps its semantics for
  back-compat; a new generic `is_base64_media_url` covers all media data URIs.

## 10. Follow-ups (not in this branch)

- **JS parity:** `blobUploader` on `@arizeai/openinference-core` TraceConfig + a masking
  rule beside `maskLongBase64ImageRule`; destination URI computable synchronously from the
  content hash while the upload promise runs in a bounded queue.
- **openai-agents realtime migration:** `_realtime.py` has private audio env vars
  (`OPENINFERENCE_HIDE_INPUT_AUDIO`, `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH`) and custom
  `input.audio.url` attribute names; migrate onto the core config fields and
  `message_content.audio` conventions.
- **Image externalization inside `input.value`:** images in `input.value` still redact
  (only the structured attribute externalizes); unify by extending the image redaction
  pass to accept the uploader.
- **Remaining OpenAI TODOs:** `image_generation_call` output, code-interpreter file
  outputs, speech/transcription endpoints.
- **Other instrumentors:** litellm image-gen `b64_json`, Anthropic document blocks,
  Google GenAI inline_data.
- **Consumer support (Phoenix/Arize):** audio rendering from `message_content.audio`,
  URI resolution/signing for `s3://`/`gs://` schemes — explicitly out of scope here.
- **Upstream convergence:** if/when OTel stabilizes upload env vars or a registry-blessed
  reference convention, alias `OPENINFERENCE_BLOB_UPLOAD_*` accordingly.

## 11. References

OTel GenAI semantic conventions:

- [semantic-conventions-genai repo][genai-repo] — dedicated home of the GenAI semconv
  (spans, metrics, events, MCP, provider docs); the old location in
  `open-telemetry/semantic-conventions` now points here
  ([moved notice](https://opentelemetry.io/docs/specs/semconv/gen-ai/)).
- [`docs/gen-ai/gen-ai-spans.md`][genai-spans] — `gen_ai.input.messages` /
  `gen_ai.output.messages` definitions and the "store content externally and record
  references on the spans" recording mode.
- [`model/gen-ai/gen-ai-input-messages.json`][genai-input-schema] and
  [`gen-ai-output-messages.json`][genai-output-schema] — normative JSON Schemas for
  message parts (`TextPart`, `ToolCallRequestPart`, `ReasoningPart`, `BlobPart`,
  `UriPart`, `FilePart`, `Modality`).
- [semantic-conventions#1556][semconv-issue-1556] (multimodal content issue) and
  [semantic-conventions#2754][semconv-pr-2754] (PR adding `blob`/`uri`/`file` parts).
- [semantic-conventions#1521][semconv-pr-1521] — proposed generic `*.blob_ref.*`
  convention; closed unmerged (why `*_ref` attribute names remain
  implementation-defined).
- [semantic-conventions v1.41.1 gen-ai docs][semconv-v1411] — source of the JSON schemas
  vendored in `openinference-instrumentation/tests/fixtures/genai_schemas/`.

OTel GenAI reference implementations:

- [opentelemetry-python-genai repo][python-genai-repo] — GenAI instrumentations
  (`opentelemetry-instrumentation-genai-openai` etc.) and `util/opentelemetry-util-genai`.
- [`opentelemetry-util-genai` on PyPI][util-genai-pypi] — experimental package defining
  `CompletionHook`, capture-mode env vars
  (`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`), and the upload configuration
  (`OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH`, `..._UPLOAD_FORMAT`,
  `..._UPLOAD_MAX_QUEUE_SIZE`).
- [`UploadCompletionHook` source][upload-hook-src] — fsspec-based whole-payload uploader
  that stamps `gen_ai.input.messages_ref` / `gen_ai.output.messages_ref` span attributes.
- [opentelemetry-python-contrib#3065][contrib-3065] — Google's `BlobUploader` proposal
  (`Blob`, `upload_async(blob) -> uri`, `BLOB_UPLOAD_URI_PREFIX`), the precedent for the
  synchronous-URI/asynchronous-write shape adopted here.

Storage layer:

- [fsspec documentation](https://filesystem-spec.readthedocs.io/) — the filesystem
  abstraction behind `FsspecBlobUploader` (and OTel's `UploadCompletionHook`);
  [built-in and third-party backend registry](https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations)
  lists the scheme → driver mapping (`s3fs`, `gcsfs`, `adlfs`, …).

[genai-repo]: https://github.com/open-telemetry/semantic-conventions-genai
[genai-spans]: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-spans.md
[genai-input-schema]: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/gen-ai-input-messages.json
[genai-output-schema]: https://github.com/open-telemetry/semantic-conventions-genai/blob/main/model/gen-ai/gen-ai-output-messages.json
[semconv-issue-1556]: https://github.com/open-telemetry/semantic-conventions/issues/1556
[semconv-pr-2754]: https://github.com/open-telemetry/semantic-conventions/pull/2754
[semconv-pr-1521]: https://github.com/open-telemetry/semantic-conventions/pull/1521
[semconv-v1411]: https://github.com/open-telemetry/semantic-conventions/tree/v1.41.1/docs/gen-ai
[python-genai-repo]: https://github.com/open-telemetry/opentelemetry-python-genai
[util-genai-pypi]: https://pypi.org/project/opentelemetry-util-genai/
[upload-hook-src]: https://github.com/open-telemetry/opentelemetry-python-genai/tree/main/util/opentelemetry-util-genai
[contrib-3065]: https://github.com/open-telemetry/opentelemetry-python-contrib/issues/3065
