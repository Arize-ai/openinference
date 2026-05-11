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

### Content Type Attributes

Each content item has a `type` attribute that identifies its kind:
- `"text"` - Text content
- `"image"` - Image content
- `"audio"` - Audio content
- `"document"` - Document content such as a PDF

### Text Content

```
llm.input_messages.0.message.contents.0.message_content.type = "text"
llm.input_messages.0.message.contents.0.message_content.text = "What is in this image?"
```

### Image Content

Image content blocks use the single-nested `message_content.image.url` shape along with a MIME type:

```
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.image.url = "https://example.com/image.jpg"
llm.input_messages.0.message.contents.1.message_content.image.mime_type = "image/jpeg"
```

The previous double-nested form `message_content.image.image.url` is **deprecated** but still accepted. New instrumentors should emit the single-nested shape.

### Audio Content

Audio content blocks use `message_content.audio.url`, with an optional transcript and an optional provider-defined voice identifier:

```
llm.input_messages.0.message.contents.2.message_content.type = "audio"
llm.input_messages.0.message.contents.2.message_content.audio.url = "https://example.com/audio.mp3"
llm.input_messages.0.message.contents.2.message_content.audio.mime_type = "audio/mpeg"
llm.input_messages.0.message.contents.2.message_content.audio.transcript = "Hello, how are you?"
```

For assistant audio outputs, include `voice_id` when known:
```
llm.output_messages.0.message.contents.0.message_content.type = "audio"
llm.output_messages.0.message.contents.0.message_content.audio.url = "gs://voice-bucket/turn-1-out.mp3"
llm.output_messages.0.message.contents.0.message_content.audio.mime_type = "audio/mpeg"
llm.output_messages.0.message.contents.0.message_content.audio.transcript = "Hello! What's your confirmation number?"
llm.output_messages.0.message.contents.0.message_content.audio.voice_id = "marin"
```

The previous double-nested form `message_content.audio.audio.url` is **deprecated** but still accepted. New instrumentors should emit the single-nested shape. The span-level `audio.url` / `audio.mime_type` / `audio.transcript` attributes are likewise deprecated in favor of the content-block-level attributes shown above.

### Document Content

Document content blocks (PDFs and similar) use `message_content.document.url` along with a MIME type and an optional filename:

```
llm.input_messages.0.message.contents.1.message_content.type = "document"
llm.input_messages.0.message.contents.1.message_content.document.url = "gs://docs-bucket/policies/travel-policy.pdf"
llm.input_messages.0.message.contents.1.message_content.document.mime_type = "application/pdf"
llm.input_messages.0.message.contents.1.message_content.document.name = "travel-policy.pdf"
```

## Provider File IDs

Some providers (OpenAI Files API, Anthropic Files API, Gemini File API, and similar) expose opaque file IDs that reference assets stored on the provider's side. Instrumentors MUST drop these file IDs rather than emit them — they should not be placed in the `url` field (which would mis-type the value) nor under a vendor namespace. When a provider call is made with a file ID and no fetchable URL is available, emit the content block with `mime_type` (and `transcript`, where applicable) and omit `url`. A future revision of this spec may introduce dedicated `file_id` fields; until then, dropping is the expected behavior.

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
  "llm.input_messages.0.message.contents.1.message_content.image.url": "https://example.com/photo.jpg",
  "llm.input_messages.0.message.contents.1.message_content.image.mime_type": "image/jpeg"
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
