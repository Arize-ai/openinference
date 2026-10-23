# Multimodal Attributes

This document describes how message content arrays represent multimodal content (text, images, audio, video, and documents such as PDFs or inline text) in OpenInference spans. The same `message.contents` structure is also used for reasoning and provider-native tool-use parts when item ordering must be preserved.

Vendor mappings live in [`internal_docs/specs/audio_video/`](../internal_docs/specs/audio_video/vendor_comparison.md). openai-agents realtime still emits instrumentor-local `input.audio.*` and `output.audio.*` strings. Those keys are not published `SpanAttributes` yet.

## Message Content Arrays

When a message contains multiple content items (e.g., text and images), the content is represented using the `message.contents` array structure with flattened attributes.

### Attribute Pattern

`llm.input_messages.<messageIndex>.message.contents.<contentIndex>.message_content.<attribute>`

Where:
- `<messageIndex>` is the zero-based index of the message
- `<contentIndex>` is the zero-based index of the content item within the message
- `<attribute>` is the specific content attribute

### Content Type Attributes

Each content item has a `type` attribute that identifies its kind:
- `"text"` - Text content
- `"image"` - Image content (URL or base64)
- `"audio"` - Audio content (URL or base64)
- `"video"` - Video content (URL or base64)
- `"document"` - Document content such as a PDF (URL or base64), an uploaded file, or inline text
- `"reasoning"` - Reasoning or thinking content, including visible summaries and Anthropic `redacted_thinking`
- `"tool_use"` - Provider-native tool-use part when a tool call must remain ordered relative to adjacent content items

Reasoning-specific fields such as `message_content.id`, `message_content.signature`, `message_content.data`, and `message_content.encrypted_content` are defined in [LLM Spans](./llm_spans.md#reasoning-content).

### Text Content

```
llm.input_messages.0.message.contents.0.message_content.type = "text"
llm.input_messages.0.message.contents.0.message_content.text = "What is in this image?"
```

### Image Content

```
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.image.image.url = "https://example.com/image.jpg"
```

For base64-encoded images:
```
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.image.image.url = "data:image/png;base64,iVBORw0KGgo..."
```

### Audio Content

```
llm.input_messages.0.message.contents.2.message_content.type = "audio"
llm.input_messages.0.message.contents.2.message_content.audio.audio.url = "https://example.com/audio.mp3"
llm.input_messages.0.message.contents.2.message_content.audio.audio.transcript = "Hello, how are you?"
```

`audio.transcript` is optional. Emit it when a transcription is available on the same part. Do not emit `audio.mime_type`. Consumers infer MIME type from the URL path extension (`.wav` to `audio/wav`, `.mp3` to `audio/mpeg`) or from a data URI prefix.

For OpenAI Chat Completions `input_audio`, build a data URI from base64 `data` and `format` (`wav` maps to `audio/wav`, `mp3` maps to `audio/mpeg`) and store that URI in `audio.audio.url`. Assistant `message.audio` on the same API is still a chat message. Put it on `llm.output_messages` audio content items.

### Video Content

```
llm.input_messages.0.message.contents.3.message_content.type = "video"
llm.input_messages.0.message.contents.3.message_content.video.video.url = "gs://bucket/clip.mp4"
```

For base64-encoded video:

```
llm.input_messages.0.message.contents.3.message_content.type = "video"
llm.input_messages.0.message.contents.3.message_content.video.video.url = "data:video/mp4;base64,AAAA..."
```

Do not emit a MIME type attribute for video. Consumers infer MIME type from the URL path extension (`.mp4` to `video/mp4`, `.webm` to `video/webm`, `.ogv` to `video/ogg`, `.avi` to `video/x-msvideo`) or from a data URI prefix. Copy `gs://`, `s3://`, and `https://` URIs verbatim. Do not put video in `image.url`. Do not store provider file ids in `video.url`.

The doubled prefix (`message_content.video.video.url`) matches `message_content.image.image.url`. It is the concatenation of `message_content.video` and `video.url`.

### Document Content

Documents passed to a model, such as PDFs or inline text documents, use `message_content.type` `"document"`:

```
llm.input_messages.0.message.contents.4.message_content.type = "document"
llm.input_messages.0.message.contents.4.message_content.document.document.url = "https://example.com/report.pdf"
llm.input_messages.0.message.contents.4.message_content.document.document.mime_type = "application/pdf"
llm.input_messages.0.message.contents.4.message_content.document.document.name = "report.pdf"
```

For base64-encoded documents:

```
llm.input_messages.0.message.contents.4.message_content.type = "document"
llm.input_messages.0.message.contents.4.message_content.document.document.url = "data:application/pdf;base64,JVBERi0xLjcK..."
llm.input_messages.0.message.contents.4.message_content.document.document.mime_type = "application/pdf"
llm.input_messages.0.message.contents.4.message_content.document.document.name = "Q3 report"
```

For a document uploaded with a provider Files API:

```
llm.input_messages.0.message.contents.4.message_content.type = "document"
llm.input_messages.0.message.contents.4.message_content.document.document.id = "file-abc123"
```

For an inline text document:

```
llm.input_messages.0.message.contents.4.message_content.type = "document"
llm.input_messages.0.message.contents.4.message_content.document.document.content = "Clause 1. The tenant pays rent monthly."
llm.input_messages.0.message.contents.4.message_content.document.document.mime_type = "text/plain"
```

A document part carries its payload in exactly one of three leaves:

- `document.url` for bytes and URIs. It may be an absolute URI (`https://`, `s3://`, `gs://`, ...) or a base64 data URI. Copy URIs verbatim. Build a data URI from provider base64 payloads. A data URI uses standard base64 (RFC 4648 section 4): re-encode URL-safe base64, which the Gemini REST API sends, and encode raw bytes from SDK objects. Text files sent as bytes (for example Bedrock `txt` or `md` documents) also go here, with their MIME type.
- `document.content` for inline text that the provider receives as a string (Anthropic `text` sources, Bedrock `text` sources, LangChain `text-plain` blocks). Do not wrap it in a data URI. Record the string byte-for-byte: no trimming, no line-ending normalization, and no truncation. Provider citations index into this string by character offset (Anthropic `char_location` and Bedrock `documentChar` give start and end positions; Anthropic `cited_text` equals the slice between them, trailing whitespace included), so any change shifts or invalidates them. If the text must be hidden or is too large to record, replace the whole value with `"__REDACTED__"` or omit it. Never record a shortened prefix.
- `document.id` for a file uploaded through a provider Files API: the provider's file id (OpenAI, LiteLLM, and LangChain `file_id`, Anthropic `file` sources). Never put a file id in `document.url`.

The other leaves describe the document:

- Emit `document.mime_type` whenever the request states the MIME type: an explicit provider field, the data URI prefix, a Bedrock `format` when present, or a source type that admits one MIME type (Anthropic `url` sources are PDF-only, Anthropic `text` sources are `text/plain`). Copy it from the request; never infer it by inspecting the document. Omit it when the request does not say, for example an OpenAI Responses `file_url` or a Bedrock document without `format`. Unlike audio and video, document URLs often have no file extension (signed URLs, object keys), so consumers cannot reliably infer the type.
- `document.name` is the name supplied with the document, copied verbatim: a file name (OpenAI and LiteLLM `filename`, LangChain `extras.filename`) or a title (Anthropic `title`, Bedrock `name`, Gemini `display_name`). Providers send at most one of these per document. It is not a unique identifier.
- Documents built from nested content blocks, such as Anthropic and Bedrock `content` sources, are not document content items. Record them with the instrumentor's generic content-block fallback.
- Do not put documents in `image.url`.
- `document.score` and `document.metadata` are retrieval leaves and are not emitted on message content.
- Model responses do not return documents today. Provider outputs reference input documents through citations (for example Anthropic `page_location` with `document_index`). Documents appear on output only for non-`LLM` spans, through `output.documents`.

Provider mappings for input document parts:

| Provider part | `document.url` | `document.content` | `document.mime_type` | `document.id` | `document.name` |
|---|---|---|---|---|---|
| OpenAI Chat Completions, Azure OpenAI, LiteLLM: `{"type": "file", "file": {...}}` | `file.file_data` (already a data URI) |  | data URI prefix, or LiteLLM `file.format` | `file.file_id` | `file.filename` |
| OpenAI Responses, openai-agents: `{"type": "input_file", ...}`, including `ToolOutputFileContent` in `function_call_output` | `file_data` (already a data URI) or `file_url` |  | data URI prefix; omit for `file_url` | `file_id` | `filename` |
| Anthropic, Bedrock InvokeModel: `{"type": "document", "source": {"type": "base64", ...}}` | data URI from `source.data` |  | `source.media_type` |  | `title` |
| Anthropic: `source.type` `"url"` | `source.url` |  | `application/pdf` |  | `title` |
| Anthropic: `source.type` `"text"` |  | `source.data` | `source.media_type` (`text/plain`) |  | `title` |
| Anthropic: `source.type` `"file"` |  |  |  | `source.file_id` | `title` |
| Bedrock Converse: `{"document": {"format": ..., "source": {...}}}` | data URI from `source.bytes`, or `source.s3Location.uri` | `source.text` | from `format` (`pdf` maps to `application/pdf`, `txt` to `text/plain`, `md` to `text/markdown`) |  | `name` |
| Gemini: `inline_data` / `inlineData` | data URI from `data` |  | `mime_type` |  | `display_name` |
| Gemini: `file_data` / `fileData` | `file_uri` (`gs://` or a Files API `https://` URI) |  | `mime_type` |  | `display_name` |
| LangChain `FileContentBlock` (`{"type": "file", ...}`) | data URI from `base64`, or `url` |  | `mime_type` | `file_id` | `extras.filename` |
| LangChain `PlainTextContentBlock` (`{"type": "text-plain", ...}`) |  | `text` | `mime_type` (`text/plain`) |  |  |

The doubled prefix (`message_content.document.document.url`) is the concatenation of `message_content.document` and `document.url`, matching `message_content.image.image.url`.

## External Storage for Large Media

Inline base64 payloads can exceed OTLP message limits and inflate backend storage. As an **experimental** capability, instrumentations MAY externalize oversized images at capture time: upload the decoded bytes to configured blob storage and record the destination URI in the same `image.image.url` attribute where the data URI would have been recorded.

```
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.image.image.url = "s3://my-bucket/oi-media/3a7bd3e2....png"
```

The same applies to output-message images (`llm.output_messages.*.message.contents.*.message_content.image.image.url`).

Semantics:
- Externalization applies only to base64 data URIs exceeding `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH`. Small payloads stay inline.
- The recorded value MUST be a valid absolute URI (a scheme is required), and SHOULD be the most consumer-resolvable form available — an `https://` or signed URL where possible; storage-scheme URIs (`gs://`, `s3://`) are valid canonical references but require viewer-side resolution. Invalid values are replaced with `"__REDACTED__"`.
- The destination URI SHOULD be content-addressed (e.g. keyed by the SHA-256 of the decoded bytes) with a mime-derived file extension, so identical content deduplicates and the URI can be computed before the upload completes.
- If no uploader is configured or the upload cannot be accepted, the existing redaction behavior applies (`"__REDACTED__"`).
- Hide settings (`OPENINFERENCE_HIDE_INPUT_IMAGES`) take precedence over externalization: hidden content is never uploaded.
- Consumers are responsible for dereferencing: URIs are not guaranteed to be publicly resolvable, and a consumer without access SHOULD treat the value as it would `"__REDACTED__"`.

This maps directly onto the OTel GenAI semantic conventions message model: an inline data URI corresponds to a `blob` part, while an externalized reference corresponds to a `uri` part. The same blob/uri split will apply to `message_content.audio.audio.url` and `message_content.video.video.url` (`modality` `audio` or `video`) when GenAI dual-write covers those parts. Shared `TraceConfig.mask()` does not yet hide or size-gate audio or video.

## Span-Kind-Independent Images

`llm.input_messages` / `llm.output_messages` carry a full chat-message structure, so they only make
sense on an `LLM` span. Other span kinds handle images too — a `TOOL` span running OCR, a `CHAIN`
step holding a browser screenshot, an image-generation call with no chat messages at all. Those
spans record images with the top-level `input.images` / `output.images` attributes. These work like
`input.value` / `output.value`: they are valid on any span kind and carry no role or message
structure.

### Attribute Pattern

`<input|output>.images.<imageIndex>.image.<attribute>`

Where:
- `<imageIndex>` is the zero-based index of the image
- `<attribute>` is `url`

For example, a tool span that receives a page scan and returns an annotated version:

```json
{
  "openinference.span.kind": "TOOL",
  "input.images.0.image.url": "data:image/png;base64,iVBORw0KGgo...",
  "output.images.0.image.url": "https://example.com/annotated.png"
}
```

Multiple images are indexed:

```
input.images.0.image.url = "https://example.com/page-1.png"
input.images.1.image.url = "https://example.com/page-2.png"
```

### Semantics

- `image.url` means the same thing as `message_content.image.image.url`: a link to the image or its
  base64 encoding. It may be an absolute URI (`https://`, `s3://`, `gs://`, ...), a base64 data URI
  (`data:image/png;base64,...`), or a bare base64 payload.
- An `LLM` span records its images inside the messages (`message.contents`). It SHOULD NOT repeat
  them under `input.images` / `output.images`.
- Images are listed in the order they occur, and the index is the only structure. There is no role
  and no link to text or to message indices. When the position of an image relative to text
  matters, use `message.contents` on an `LLM` span instead.
- `input.value` / `output.value` are unaffected. They remain the textual input and output of the
  span.

### Redaction and Size Limits

The image privacy controls apply to these attributes the same way they apply to message-content
images:

- `OPENINFERENCE_HIDE_INPUT_IMAGES` removes all `input.images.*` attributes.
- `OPENINFERENCE_HIDE_INPUTS` removes `input.images.*` and `OPENINFERENCE_HIDE_OUTPUTS` removes
  `output.images.*`, just as they hide `input.value` / `output.value`.
- `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` and a configured blob uploader apply to
  `<input|output>.images.<i>.image.url` (see
  [External Storage for Large Media](#external-storage-for-large-media)). A hidden image is never
  uploaded.

Support matches what each SDK already does for message-content images: Python enforces the size
limit and can externalize to a blob uploader; JavaScript enforces the size limit and redacts
oversized images to `"__REDACTED__"`; Java and Go apply the hide flags only.

## Span-Kind-Independent Documents

Documents such as PDFs follow the same rules as
[span-kind-independent images](#span-kind-independent-images). Spans other than `LLM` record them
with the top-level `input.documents` / `output.documents` attributes, for example a `TOOL` span that
fills in a PDF form or a `CHAIN` step that receives an uploaded file.

### Attribute Pattern

`<input|output>.documents.<documentIndex>.document.<attribute>`

Where:
- `<documentIndex>` is the zero-based index of the document
- `<attribute>` is `url`, `content`, `mime_type`, `id`, or `name`

```json
{
  "openinference.span.kind": "TOOL",
  "input.documents.0.document.url": "data:application/pdf;base64,JVBERi0xLjcK...",
  "input.documents.0.document.mime_type": "application/pdf",
  "output.documents.0.document.url": "https://example.com/filled-form.pdf",
  "output.documents.0.document.mime_type": "application/pdf"
}
```

### Semantics

- Each `document.*` leaf means the same thing as under `message_content.document`
  (see [Document Content](#document-content)).
- An `LLM` span records its documents inside the messages (`message.contents`). It SHOULD NOT
  repeat them under `input.documents` / `output.documents`.
- `input.documents` / `output.documents` are distinct from `retrieval.documents` and
  `reranker.*_documents`, which list retrieved text passages and use the `document.id`,
  `document.content`, `document.score`, and `document.metadata` leaves.
- Shared `TraceConfig.mask()` does not yet hide or size-gate documents, in
  `input.documents` / `output.documents` or in `message_content.document`. Inline base64 PDFs can
  exceed attribute and OTLP payload limits. Hide flags and a size limit are a planned follow-up.

## Privacy Considerations

### Hiding Images

When `OPENINFERENCE_HIDE_INPUT_IMAGES` is set to true:
- Image URLs in input messages will be replaced with `"__REDACTED__"`
- This only applies when input messages are not already completely hidden

### Base64 Image Truncation

When `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` is set (default: 32000):
- Base64-encoded images longer than this limit will be truncated
- The truncation preserves the data URL prefix (e.g., `data:image/png;base64,`)
- Only the base64 content portion is subject to the length limit
- If a blob uploader is configured, over-limit images are externalized instead and the attribute records the destination URI (see [External Storage for Large Media](#external-storage-for-large-media))

### Hiding Text Content

When `OPENINFERENCE_HIDE_INPUT_TEXT` is set to true:
- Text content in multimodal messages will be replaced with `"__REDACTED__"`
- This only applies when input messages are not already completely hidden

## Example: Multimodal Message

A user message with both text and image content:

```json
{
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.contents.0.message_content.text": "What objects do you see in this image?",
  "llm.input_messages.0.message.contents.1.message_content.type": "image",
  "llm.input_messages.0.message.contents.1.message_content.image.image.url": "https://example.com/photo.jpg"
}
```

## Fallback for Simple Messages

When a message contains only text content (no multimodal content), it can use the simpler format:

```json
{
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.0.message.content": "Hello, how are you?"
}
```

The `message.content` attribute is used for simple text-only messages, while `message.contents` is used for multimodal messages.
