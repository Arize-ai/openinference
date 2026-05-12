# Multimodal Attributes

This document describes how multimodal content (text, images, audio, documents) is represented in OpenInference spans.

## Message Content Arrays

When a message contains multiple content items (e.g., text and images), the content is represented using the `message.contents` array structure with flattened attributes.

### Attribute Pattern

`llm.input_messages.<messageIndex>.message.contents.<contentIndex>.message_content.<attribute>`

Where:
- `<messageIndex>` is the zero-based index of the message
- `<contentIndex>` is the zero-based index of the content item within the message
- `<attribute>` is the specific content attribute

### Content Type

Each content item has a required `type` attribute that identifies its semantic category. `type` is the discriminator used by renderers, redaction flags, and conditional-field rules. `mime_type` (when present) gives the precise wire format within a category.

The set of well-known `type` values is **closed**; adding new values requires a spec update.

- `"text"` — Text content
- `"image"` — Image content
- `"audio"` — Audio content
- `"document"` — Document content such as a PDF

### Text Content

Text content is special-cased: it lives directly on `message_content.text` rather than under a `file.*` namespace.

```
llm.input_messages.0.message.contents.0.message_content.type = "text"
llm.input_messages.0.message.contents.0.message_content.text = "What is in this image?"
```

### File Content (Image, Audio, Document)

All non-text content types share a single `message_content.file.*` namespace. Renderers, redaction flags, and conditional-field rules branch on `type`; the shared shape removes the need for per-type duplication.

**Shared fields** (apply to any `type` other than `"text"`):

- `file.url` — fetchable URL or a `data:` URI carrying inline base64 bytes
- `file.mime_type` — optional; recommended for `data:` URIs and any URL whose extension does not unambiguously identify the format
- `file.name` — optional filename or label
- `file.file_id` — optional opaque provider file ID; used when no fetchable URL is available

**Type-conditional fields**:

- `file.transcript` — applies when `type = "audio"`; rendered text of the audio content (input or assistant output)

The TTS voice preset is request-side configuration rather than content metadata; emit it as `llm.voice_name` (an invocation parameter) on the span rather than on individual content blocks.

#### Image

```
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.file.url = "https://example.com/image.jpg"
llm.input_messages.0.message.contents.1.message_content.file.mime_type = "image/jpeg"
```

Inline base64 bytes use a `data:` URI as the `url`:

```
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.file.url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
llm.input_messages.0.message.contents.1.message_content.file.mime_type = "image/png"
```

#### Audio

```
llm.input_messages.0.message.contents.2.message_content.type = "audio"
llm.input_messages.0.message.contents.2.message_content.file.url = "https://example.com/audio.mp3"
llm.input_messages.0.message.contents.2.message_content.file.mime_type = "audio/mpeg"
llm.input_messages.0.message.contents.2.message_content.file.transcript = "Hello, how are you?"
```

For assistant audio outputs, the same shape is used; emit `llm.voice_name` on the span if a specific TTS preset was requested:

```
llm.voice_name = "alloy"

llm.output_messages.0.message.contents.0.message_content.type = "audio"
llm.output_messages.0.message.contents.0.message_content.file.url = "gs://voice-bucket/turn-1-out.mp3"
llm.output_messages.0.message.contents.0.message_content.file.mime_type = "audio/mpeg"
llm.output_messages.0.message.contents.0.message_content.file.transcript = "Hello! What's your confirmation number?"
```

#### Document

Documents today cover PDFs; other document MIME types are expected to use the same shape as they're adopted.

```
llm.input_messages.0.message.contents.1.message_content.type = "document"
llm.input_messages.0.message.contents.1.message_content.file.url = "gs://docs-bucket/policies/travel-policy.pdf"
llm.input_messages.0.message.contents.1.message_content.file.mime_type = "application/pdf"
llm.input_messages.0.message.contents.1.message_content.file.name = "travel-policy.pdf"
```

## Deprecated Shapes

The following pre-existing attribute shapes are **deprecated** but still accepted by consumers for backwards compatibility. New instrumentors MUST emit the `message_content.file.*` shape with the appropriate `type` value.

- `message_content.image.image.url` (double-nested image URL) — emit `message_content.file.url` with `type = "image"` instead.
- Span-level `audio.url`, `audio.mime_type`, `audio.transcript` — emit the corresponding `message_content.file.*` fields on a content block with `type = "audio"` instead.

## Provider File IDs

Some providers (OpenAI Files API, Anthropic Files API, Gemini File API, and similar) expose opaque file IDs that reference assets stored on the provider's side. These are not fetchable URLs and SHOULD NOT be placed in `file.url`. Emit `file.file_id` instead, along with any other known metadata (`mime_type`, `name`, `transcript`):

```
llm.input_messages.0.message.contents.0.message_content.type = "document"
llm.input_messages.0.message.contents.0.message_content.file.file_id = "file-abc123XYZ"
llm.input_messages.0.message.contents.0.message_content.file.mime_type = "application/pdf"
llm.input_messages.0.message.contents.0.message_content.file.name = "termination-letter.pdf"
```

## Privacy Considerations

Multimodal content can carry sensitive payloads (PII in audio transcripts, document URLs that leak filenames, etc.). Each non-text content type gets a single redaction flag — when set, *all* fields of matching content blocks are redacted (`file.url`, `file.mime_type`, `file.name`, `file.file_id`, `file.transcript`) rather than picking individual sub-fields. Redaction applies only when the surrounding input messages are not already completely hidden.

### Hiding Images

When `OPENINFERENCE_HIDE_INPUT_IMAGES` is set to true:
- All fields of input content blocks with `type = "image"` will be replaced with `"__REDACTED__"`
- This only applies when input messages are not already completely hidden

### Hiding Audio

When `OPENINFERENCE_HIDE_INPUT_AUDIO` is set to true:
- All fields of input content blocks with `type = "audio"` will be replaced with `"__REDACTED__"`
- The transcript is particularly sensitive because it *is* the rendered message content for voice conversations
- This only applies when input messages are not already completely hidden

### Hiding Documents

When `OPENINFERENCE_HIDE_INPUT_DOCUMENTS` is set to true:
- All fields of input content blocks with `type = "document"` will be replaced with `"__REDACTED__"`
- `file.name` may itself be sensitive (e.g., filenames that disclose the subject matter)
- This only applies when input messages are not already completely hidden

### Base64 Image Truncation

When `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` is set (default: 32000):
- Base64-encoded images longer than this limit will be truncated
- The truncation preserves the data URL prefix (e.g., `data:image/png;base64,`)
- Only the base64 content portion is subject to the length limit

### Hiding Text Content

When `OPENINFERENCE_HIDE_INPUT_TEXT` is set to true:
- Text content in multimodal messages will be replaced with `"__REDACTED__"`
- This only applies when input messages are not already completely hidden

## Example: Audio-to-Audio Conversation Turn

Voice agents typically observe a single round of conversation as one LLM span whose input *and* output are both audio. The `transcript` field on each block carries the rendered text, while `url` carries the audio asset; backends that cannot replay audio can still search and display the conversation via the transcripts. The requested TTS voice preset lives on the span as `llm.voice_name`.

```json
{
  "llm.voice_name": "alloy",

  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.0.message.contents.0.message_content.type": "audio",
  "llm.input_messages.0.message.contents.0.message_content.file.url": "gs://voice-bucket/turn-1-in.wav",
  "llm.input_messages.0.message.contents.0.message_content.file.mime_type": "audio/wav",
  "llm.input_messages.0.message.contents.0.message_content.file.transcript": "What's the weather today?",

  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.contents.0.message_content.type": "audio",
  "llm.output_messages.0.message.contents.0.message_content.file.url": "gs://voice-bucket/turn-1-out.mp3",
  "llm.output_messages.0.message.contents.0.message_content.file.mime_type": "audio/mpeg",
  "llm.output_messages.0.message.contents.0.message_content.file.transcript": "It's sunny and 72°F."
}
```

A turn cut off by the user emits the same shape, plus `openinference.end_reason = "interrupted_by_user"` on the span. The assistant `transcript` should reflect only what was actually synthesized before the interruption.

## Example: Multimodal Message

A user message with both text and image content:

```json
{
  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.0.message.contents.0.message_content.type": "text",
  "llm.input_messages.0.message.contents.0.message_content.text": "What objects do you see in this image?",
  "llm.input_messages.0.message.contents.1.message_content.type": "image",
  "llm.input_messages.0.message.contents.1.message_content.file.url": "https://example.com/photo.jpg",
  "llm.input_messages.0.message.contents.1.message_content.file.mime_type": "image/jpeg"
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
