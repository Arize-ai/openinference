# Blob Upload

**This capability is experimental** — the uploader contract and attribute semantics may change while the feature
matures.

Large binary content captured as base64 data URIs can exceed span attribute and OTLP payload limits and inflate
backend storage. Instead of redacting oversized media, an instrumentation MAY upload the decoded bytes to external
storage at capture time and record only a reference URI in the span attribute, in the same attribute where the data
URI would have been recorded.

## Scope

Blob upload applies to **images** and **documents** whose base64 data URI exceeds
`OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH`:

- `message_content.image.image.url` in input and output messages, and `<input|output>.images.*.image.url`
- `message_content.document.document.url` in input and output messages, and `<input|output>.documents.*.document.url`

Audio and video data URIs are not size-gated.

## Attribute Semantics

```
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.image.image.url = "s3://my-bucket/oi-media/3a7bd3e2....png"
llm.input_messages.0.message.contents.2.message_content.document.document.url = "s3://my-bucket/oi-media/9f86d081....pdf"
```

- Externalization applies only to base64 data URIs exceeding `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH`. Small payloads
  stay inline.
- The recorded value MUST be a valid absolute URI (a scheme is required), and SHOULD be the most consumer-resolvable
  form available — an `https://` or signed URL where possible; storage-scheme URIs (`gs://`, `s3://`) are valid
  canonical references but require viewer-side resolution. Invalid values are replaced with `"__REDACTED__"`.
- The destination URI SHOULD be content-addressed (e.g. keyed by the SHA-256 of the decoded bytes) with a mime-derived
  file extension, so identical content deduplicates and the URI can be computed before the upload completes.
- If no uploader is configured or the upload cannot be accepted, the existing redaction behavior applies
  (`"__REDACTED__"`).
- Hide settings (for example `OPENINFERENCE_HIDE_INPUTS`, `OPENINFERENCE_HIDE_INPUT_MESSAGES`,
  `OPENINFERENCE_HIDE_INPUT_IMAGES`) take precedence over externalization: hidden content is never uploaded.
- Consumers are responsible for dereferencing: URIs are not guaranteed to be publicly resolvable, and a consumer
  without access SHOULD treat the value as it would `"__REDACTED__"`.

This maps directly onto the OTel GenAI semantic conventions message model: an inline data URI corresponds to a `blob`
part, while an externalized reference corresponds to a `uri` part. The same blob/uri split will apply to
`message_content.audio.audio.url` and `message_content.video.video.url` (`modality` `audio` or `video`) when GenAI
dual-write covers those parts.

## Configuring an Uploader

OpenInference defines the interface and the offload policy but ships no uploader implementation — implementations
come from applications, vendor SDKs (e.g. the Arize SDK), or a future upstream (OTel util-genai) byte uploader. In
Python an uploader is supplied either in code, or zero-code via an entry
point:

```python
from openinference.instrumentation import TraceConfig

config = TraceConfig(blob_uploader=my_uploader)  # any object satisfying BlobUploader
```

```toml
# the package providing the uploader registers it in its own packaging
# metadata under the "openinference_blob_uploader" entry-point group:
[project.entry-points.openinference_blob_uploader]
arize = "arize_otel.blob:ArizeBlobUploader"
```

```bash
export OPENINFERENCE_BLOB_UPLOADER=arize
```

The name is resolved when a `TraceConfig` is constructed: the entry point is imported, instantiated if it is a class or
zero-argument factory, and validated against the `BlobUploader` protocol. Loads are memoized per name, so all
instrumentors in a process share a single uploader instance. Resolution failures log a warning and leave the uploader
unset — oversized media then redacts exactly as with no uploader configured. A `blob_uploader` passed in code takes
precedence over the environment variable (and is validated at construction: passing a string or a class raises a
`TypeError`). The uploader's own configuration (bucket, credentials, endpoint) is read by the uploader itself,
typically from its own environment variables (e.g. a base URL), so fully zero-code deployments stay possible.

## Uploader Contract

Implementations MUST NOT block the instrumented call: return the destination URI immediately (content-addressed
naming, e.g. SHA-256 of the bytes, makes it computable before any I/O) and move the bytes on a background worker. The
returned value MUST be a valid absolute URI with a scheme, and SHOULD be the most consumer-resolvable form available
(an `https://` or signed URL where possible; storage-scheme URIs such as `gs://` are valid canonical references but
require viewer-side resolution) — the caller validates the returned value and redacts invalid ones. Returning `None`
(backpressure, shutdown, policy) makes the caller fall back to the standard redaction behavior. Implementations own
their lifecycle: flush pending uploads at process exit (e.g. an `atexit` hook), following the `BatchSpanProcessor`
precedent of the worker-owner owning shutdown.
