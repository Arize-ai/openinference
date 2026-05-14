# Multimodal Attributes

This document describes how multimodal content (text, images, audio) is represented in OpenInference spans.

## Message Content Arrays

When a message contains multiple content items (e.g., text and audio), the content is represented using the `message.contents` array structure with flattened attributes.

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

Audio content blocks use the `message_content.audio.*` namespace:

- `message_content.audio.url` — fetchable URL or a `data:` URI carrying inline base64 bytes
- `message_content.audio.mime_type` — optional; recommended for `data:` URIs and any URL whose extension does not unambiguously identify the format
- `message_content.audio.transcript` — rendered text of the audio content; optional on input audio, recommended on assistant audio outputs

```
llm.input_messages.0.message.contents.0.message_content.type = "audio"
llm.input_messages.0.message.contents.0.message_content.audio.url = "https://example.com/audio.mp3"
llm.input_messages.0.message.contents.0.message_content.audio.mime_type = "audio/mpeg"
llm.input_messages.0.message.contents.0.message_content.audio.transcript = "Hello, how are you?"
```

For inline base64 bytes, use a `data:` URI as the `url`:

```
llm.input_messages.0.message.contents.0.message_content.type = "audio"
llm.input_messages.0.message.contents.0.message_content.audio.url = "data:audio/wav;base64,UklGRiQAAABXQVZF..."
llm.input_messages.0.message.contents.0.message_content.audio.mime_type = "audio/wav"
```

For assistant audio outputs the same shape is used. The requested text-to-speech (TTS) voice preset is a request-side knob and belongs inside `llm.invocation_parameters` (alongside other invocation params like `temperature`, `top_p`, etc.):

```
llm.invocation_parameters = "{\"model\": \"gpt-realtime\", \"voice\": \"alloy\"}"

llm.output_messages.0.message.contents.0.message_content.type = "audio"
llm.output_messages.0.message.contents.0.message_content.audio.url = "gs://voice-bucket/turn-1-out.mp3"
llm.output_messages.0.message.contents.0.message_content.audio.mime_type = "audio/mpeg"
llm.output_messages.0.message.contents.0.message_content.audio.transcript = "Hello! What's your confirmation number?"
```

## Privacy Considerations

Multimodal content can carry sensitive payloads (PII in audio transcripts, URLs that leak file locations, etc.). Redaction flags operate per content type and per direction (input vs. output). When set, all fields of matching content blocks are redacted rather than picking individual sub-fields. Redaction applies only when the surrounding messages are not already completely hidden.

### Hiding Images

When `OPENINFERENCE_HIDE_INPUT_IMAGES` is set to true:
- Image URLs in input messages will be replaced with `"__REDACTED__"`
- This only applies when input messages are not already completely hidden

### Hiding Audio

When `OPENINFERENCE_HIDE_INPUT_AUDIO` is set to true:
- All fields of input content blocks with `type = "audio"` (`audio.url`, `audio.mime_type`, `audio.transcript`) will be replaced with `"__REDACTED__"`
- The transcript is particularly sensitive because it *is* the rendered message content for voice conversations
- This only applies when input messages are not already completely hidden

When `OPENINFERENCE_HIDE_OUTPUT_AUDIO` is set to true:
- All fields of output content blocks with `type = "audio"` will be replaced with `"__REDACTED__"`
- Assistant audio transcripts may repeat sensitive information back to the user (account numbers, names, health information, etc.)
- This only applies when output messages are not already completely hidden

### Hiding Text Content

When `OPENINFERENCE_HIDE_INPUT_TEXT` is set to true:
- Text content in multimodal messages will be replaced with `"__REDACTED__"`
- This only applies when input messages are not already completely hidden

### Base64 Image Truncation

When `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` is set (default: 32000):
- Base64-encoded images longer than this limit will be truncated
- The truncation preserves the data URL prefix (e.g., `data:image/png;base64,`)
- Only the base64 content portion is subject to the length limit

### Base64 Audio Truncation

When `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` is set (default: 32000):
- Any `data:` URI in `message_content.audio.url` whose base64 payload exceeds this limit will be truncated
- The truncation preserves the `data:` URI prefix (e.g., `data:audio/wav;base64,`)
- Only the base64 content portion is subject to the length limit

## Example: Audio-to-Audio Conversation Turn

Voice agents typically observe a single round of conversation as one LLM span whose input *and* output are both audio. The `transcript` field on each block carries the rendered text, while `url` carries the audio asset; backends that cannot replay audio can still search and display the conversation via the transcripts. Request-side knobs like the requested TTS voice preset belong inside `llm.invocation_parameters`.

```json
{
  "llm.invocation_parameters": "{\"model\": \"gpt-realtime\", \"voice\": \"alloy\"}",

  "llm.input_messages.0.message.role": "user",
  "llm.input_messages.0.message.contents.0.message_content.type": "audio",
  "llm.input_messages.0.message.contents.0.message_content.audio.url": "gs://voice-bucket/turn-1-in.wav",
  "llm.input_messages.0.message.contents.0.message_content.audio.mime_type": "audio/wav",
  "llm.input_messages.0.message.contents.0.message_content.audio.transcript": "What's the weather today?",

  "llm.output_messages.0.message.role": "assistant",
  "llm.output_messages.0.message.contents.0.message_content.type": "audio",
  "llm.output_messages.0.message.contents.0.message_content.audio.url": "gs://voice-bucket/turn-1-out.mp3",
  "llm.output_messages.0.message.contents.0.message_content.audio.mime_type": "audio/mpeg",
  "llm.output_messages.0.message.contents.0.message_content.audio.transcript": "It's sunny and 72°F."
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
