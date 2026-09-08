# Multimodal Attributes

This document describes how message content arrays represent multimodal content (text, images, audio, video) in OpenInference spans. The same `message.contents` structure is also used for reasoning and provider-native tool-use parts when item ordering must be preserved.

Voice sessions that are not chat message lists use span-root `input.audio.*` and `output.audio.*` instead. See [Span-root audio](#span-root-audio). Vendor mappings live in [`internal_docs/specs/audio_video/`](../internal_docs/specs/audio_video/vendor_comparison.md).

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
llm.input_messages.0.message.contents.2.message_content.audio.audio.transcript = "Hello, how are you?"
```

`audio.mime_type` and `audio.transcript` are optional. Emit `mime_type` when the provider sends it. Emit `transcript` when a transcription is available on the same part.

For OpenAI Chat Completions `input_audio`, build a data URI from base64 `data` and `format` (`wav` maps to `audio/wav`, `mp3` maps to `audio/mpeg`) and store that URI in `audio.audio.url`. Assistant `message.audio` on the same API is still a chat message. Put it on `llm.output_messages` audio content items, not on span-root `output.audio.*`.

### Video Content

```
llm.input_messages.0.message.contents.3.message_content.type = "video"
llm.input_messages.0.message.contents.3.message_content.video.video.url = "gs://bucket/clip.mp4"
llm.input_messages.0.message.contents.3.message_content.video.video.mime_type = "video/mp4"
```

For base64-encoded video:

```
llm.input_messages.0.message.contents.3.message_content.type = "video"
llm.input_messages.0.message.contents.3.message_content.video.video.url = "data:video/mp4;base64,AAAA..."
llm.input_messages.0.message.contents.3.message_content.video.video.mime_type = "video/mp4"
```

`video.mime_type` is optional. Emit it when the provider sends a MIME type (Gemini `mime_type`) or when it can be derived from a format literal (Bedrock `format: "mp4"` to `video/mp4`). Copy `gs://`, `s3://`, and `https://` URIs verbatim. Do not put video in `image.url`. Do not store provider file ids in `video.url`.

The doubled prefix (`message_content.video.video.url`) matches `message_content.image.image.url`. It is the concatenation of `message_content.video` and `video.url`.

### Span-root audio

Realtime and other voice sessions that are not chat `messages[]` lists record audio on the span itself:

```
input.audio.url = "data:audio/wav;base64,..."
input.audio.mime_type = "audio/wav"
input.audio.transcript = "What's the weather in Tokyo?"
output.audio.url = "data:audio/wav;base64,..."
output.audio.mime_type = "audio/wav"
output.audio.transcript = "It's sunny in Tokyo."
```

These keys reuse `audio.url`, `audio.mime_type`, and `audio.transcript` with an `input.` or `output.` prefix. The openai-agents realtime instrumentor already emits them on USER and LLM spans. Span kinds `USER` and `AUDIO` stay instrumentor-local and are not part of this convention.

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

This maps directly onto the OTel GenAI semantic conventions message model: an inline data URI corresponds to a `blob` part, while an externalized reference corresponds to a `uri` part. The same split applies to `message_content.audio.audio.url` and `message_content.video.video.url` (`modality` `audio` or `video`). Span-root `input.audio.*` and `output.audio.*` are not dual-written until a separate mapper exists.

## Privacy Considerations

### Hiding Images

When `OPENINFERENCE_HIDE_INPUT_IMAGES` is set to true:
- Image URLs in input messages will be replaced with `"__REDACTED__"`
- This only applies when input messages are not already completely hidden

### Hiding audio and video

When `OPENINFERENCE_HIDE_INPUT_AUDIO` is true, replace `message_content.audio.audio.url` (and optional mime type and transcript) on input messages, and span-root `input.audio.*`, with `"__REDACTED__"` or drop them. `OPENINFERENCE_HIDE_OUTPUT_AUDIO` does the same for output messages and `output.audio.*`.

When `OPENINFERENCE_HIDE_INPUT_VIDEO` is true, replace `message_content.video.video.url` (and optional mime type) on input messages with `"__REDACTED__"`. `OPENINFERENCE_HIDE_OUTPUT_VIDEO` does the same for output messages.

These flags apply only when the enclosing input or output messages are not already hidden.

### Base64 Image Truncation

When `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` is set (default: 32000):
- Base64-encoded images longer than this limit will be truncated
- The truncation preserves the data URL prefix (e.g., `data:image/png;base64,`)
- Only the base64 content portion is subject to the length limit
- If a blob uploader is configured, over-limit images are externalized instead and the attribute records the destination URI (see [External Storage for Large Media](#external-storage-for-large-media))

### Base64 audio and video truncation

`OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` (default: 32000) truncates or offloads `audio.url` data URIs the same way images are truncated. Keep the `data:audio/…;base64,` prefix.

`OPENINFERENCE_BASE64_VIDEO_MAX_LENGTH` does the same for `video.url` once an instrumentor emits video data URIs. Default 32000.

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
