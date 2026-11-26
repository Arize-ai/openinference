# Multimodal Content

This document describes how message content arrays represent multimodal content (text, images, audio, video, and documents such as PDFs or inline text) in OpenInference spans. The same `message.contents` structure is also used for reasoning and provider-native tool-use parts when item ordering must be preserved.

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

Reasoning-specific fields such as `message_content.id`, `message_content.signature`, `message_content.data`, and `message_content.encrypted_content` are defined in [Reasoning Content](./messages.md#reasoning-content). A message that contains only text can use `message.content` instead; see [Text Messages](./messages.md#text-messages).

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

Documents passed to a model, such as PDFs, uploaded files, or inline text, use `message_content.type` `"document"`:

```
llm.input_messages.0.message.contents.4.message_content.type = "document"
llm.input_messages.0.message.contents.4.message_content.document.document.url = "data:application/pdf;base64,JVBERi0xLjcK..."
llm.input_messages.0.message.contents.4.message_content.document.document.mime_type = "application/pdf"
llm.input_messages.0.message.contents.4.message_content.document.document.name = "report.pdf"
```

| Leaf | Holds |
|---|---|
| `document.url` | The document bytes: a URI copied verbatim (`https://`, `gs://`, `s3://`), or a base64 data URI |
| `document.content` | Inline text the provider receives as a string |
| `document.id` | The provider file id of a file uploaded through a Files API |
| `document.mime_type` | The MIME type stated in the request |
| `document.name` | The file name or title supplied with the document; not a unique identifier |

- Set exactly one of `document.url`, `document.content`, or `document.id`. Never put a file id in `document.url`.
- Data URIs use standard base64; re-encode URL-safe base64 such as the Gemini REST API sends.
- Record `document.content` byte-for-byte, with no trimming, line-ending normalization, or truncation: provider citations index into it by character offset. To hide it, replace the whole value with `"__REDACTED__"` or omit it.
- Copy `document.mime_type` (a provider field or the data URI prefix) and `document.name` from the request; never infer them. Omit them when the request does not state them.
- Anthropic and Bedrock `content` sources (nested content blocks) are not represented by these leaves. Do not put documents in `image.url`.

Provider mappings:

| Provider part | `url` | `content` | `mime_type` | `id` | `name` |
|---|---|---|---|---|---|
| OpenAI Chat `file`, Azure OpenAI, LiteLLM | `file_data` | | data URI prefix, LiteLLM `format` | `file_id` | `filename` |
| OpenAI Responses `input_file`, openai-agents | `file_data`, `file_url` | | data URI prefix | `file_id` | `filename` |
| Anthropic `document`: `base64` / `url` / `text` / `file` source | data URI from `data`, or `url` | `data` (`text` source) | `media_type`; `application/pdf` for `url` sources | `file_id` | `title` |
| Bedrock Converse `document` | data URI from `bytes`, or `s3Location.uri` | `text` | from `format` | | `name` |
| Gemini `inline_data` / `file_data` | data URI from `data`, or `file_uri` | | `mime_type` | | `display_name` |
| LangChain `file` / `text-plain` | data URI from `base64`, or `url` | `text` | `mime_type` | `file_id` | `extras.filename` |

The doubled prefix (`message_content.document.document.url`) matches `message_content.image.image.url`.

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

The image privacy controls and size limit apply to these attributes the same way they apply to message-content images.
See [Data Masking](../configuration/data-masking.md#images-and-documents).

## Span-Kind-Independent Documents

Spans other than `LLM` record documents with `input.documents` / `output.documents`, following
[span-kind-independent images](#span-kind-independent-images). Each leaf means the same as under
[Document Content](#document-content).

`<input|output>.documents.<documentIndex>.document.<attribute>`

```json
{
  "openinference.span.kind": "TOOL",
  "input.documents.0.document.url": "data:application/pdf;base64,JVBERi0xLjcK...",
  "input.documents.0.document.mime_type": "application/pdf",
  "output.documents.0.document.url": "https://example.com/filled-form.pdf"
}
```

An `LLM` span records its documents in `message.contents` and SHOULD NOT repeat them here.

## Privacy and Size Limits

The `OPENINFERENCE_HIDE_*` settings hide content items, and `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` limits the size of
inline base64 images and documents. See [Data Masking](../configuration/data-masking.md). Oversized images and
documents can instead be uploaded to external storage; see [Blob Upload](../configuration/blob-upload.md).

## Vendor Mappings

Per-provider mappings for audio and video live in [`internal_docs/specs/audio_video/`](../../internal_docs/specs/audio_video/vendor_comparison.md). openai-agents realtime still emits instrumentor-local `input.audio.*` and `output.audio.*` strings. Those keys are not published `SpanAttributes` yet.
