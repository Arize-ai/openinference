# Multimodal Attributes

This document describes how message content arrays represent multimodal content (text, images, audio) in OpenInference spans. The same `message.contents` structure is also used for reasoning and provider-native tool-use parts when item ordering must be preserved.

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
- `"file"` - File content such as a PDF document (URL, base64, or provider file id)
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
llm.input_messages.0.message.contents.2.message_content.audio.audio.mime_type = "audio/mpeg"
```

For base64-encoded audio (e.g. OpenAI chat completions `input_audio` parts), the audio is represented as a data URI:
```
llm.input_messages.0.message.contents.2.message_content.type = "audio"
llm.input_messages.0.message.contents.2.message_content.audio.audio.url = "data:audio/wav;base64,UklGRi..."
llm.input_messages.0.message.contents.2.message_content.audio.audio.mime_type = "audio/wav"
```

Output audio (e.g. OpenAI chat completions audio responses) additionally carries the transcript when available:
```
llm.output_messages.0.message.contents.0.message_content.type = "audio"
llm.output_messages.0.message.contents.0.message_content.audio.audio.url = "data:audio/wav;base64,UklGRi..."
llm.output_messages.0.message.contents.0.message_content.audio.audio.transcript = "Hello there!"
```

### File Content

Files (e.g. PDF documents) attached to a message may arrive as inline base64, as a URL, or as a provider file id referencing content pre-uploaded to the provider (e.g. the OpenAI Files API).

```
llm.input_messages.0.message.contents.1.message_content.type = "file"
llm.input_messages.0.message.contents.1.message_content.file.file.url = "data:application/pdf;base64,JVBERi..."
llm.input_messages.0.message.contents.1.message_content.file.file.mime_type = "application/pdf"
llm.input_messages.0.message.contents.1.message_content.file.file.name = "report.pdf"
```

For a provider-managed file, only the id is recorded — no bytes are captured:
```
llm.input_messages.0.message.contents.1.message_content.type = "file"
llm.input_messages.0.message.contents.1.message_content.file.file.id = "file-abc123"
```

## External Storage for Large Media

Inline base64 payloads for audio and documents can reach tens of megabytes, exceeding OTLP message limits and inflating backend storage. As an experimental capability, instrumentations MAY externalize oversized media at capture time: upload the decoded bytes to configured blob storage and record the destination URI in the same `*.url` attribute where the data URI would have been recorded.

```
llm.input_messages.0.message.contents.2.message_content.type = "audio"
llm.input_messages.0.message.contents.2.message_content.audio.audio.url = "s3://my-bucket/oi-media/3a7bd3e2....wav"
llm.input_messages.0.message.contents.2.message_content.audio.audio.mime_type = "audio/wav"
```

Semantics:
- Externalization applies only to base64 data URIs exceeding `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` (images) or `OPENINFERENCE_BASE64_MEDIA_MAX_LENGTH` (audio, files, video). Small payloads stay inline.
- The destination URI SHOULD be content-addressed (e.g. keyed by the SHA-256 of the decoded bytes) so identical content deduplicates and the URI can be computed before the upload completes.
- If no uploader is configured or the upload cannot be accepted, the existing redaction behavior applies (`"__REDACTED__"`).
- Hide settings (`OPENINFERENCE_HIDE_INPUT_IMAGES`, `OPENINFERENCE_HIDE_INPUT_AUDIO`, `OPENINFERENCE_HIDE_OUTPUT_AUDIO`, `OPENINFERENCE_HIDE_INPUT_FILES`) take precedence over externalization: hidden content is never uploaded.

This maps directly onto the OTel GenAI semantic conventions message model: an inline data URI corresponds to a `blob` part, while an externalized reference corresponds to a `uri` part, and a provider file id corresponds to a `file` part.

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

### Hiding and Limiting Audio and Files

When `OPENINFERENCE_HIDE_INPUT_AUDIO`, `OPENINFERENCE_HIDE_OUTPUT_AUDIO`, or `OPENINFERENCE_HIDE_INPUT_FILES` is set to true, the corresponding `message_content.audio` / `message_content.file` attributes are dropped from messages.

When `OPENINFERENCE_BASE64_MEDIA_MAX_LENGTH` is set (default: 32000), base64 data URIs carrying audio or file content longer than the limit are replaced with `"__REDACTED__"` — unless a blob uploader is configured, in which case the content is externalized and the attribute is set to the destination URI (see [External Storage for Large Media](#external-storage-for-large-media)).

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
