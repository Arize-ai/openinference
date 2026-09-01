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

The optional `image.mime_type` records the type of an image whose URL does not carry it inline:
```
llm.input_messages.0.message.contents.1.message_content.type = "image"
llm.input_messages.0.message.contents.1.message_content.image.image.url = "gs://my-bucket/photo"
llm.input_messages.0.message.contents.1.message_content.image.image.mime_type = "image/png"
```

### Audio Content

```
llm.input_messages.0.message.contents.2.message_content.type = "audio"
llm.input_messages.0.message.contents.2.message_content.audio.audio.url = "https://example.com/audio.mp3"
```

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

This maps directly onto the OTel GenAI semantic conventions message model: an inline data URI corresponds to a `blob` part, while an externalized reference corresponds to a `uri` part. Audio and file content will gain the same treatment once their message-content conventions are established.

## Span-Kind-Independent Images

Images recorded through `llm.input_messages` / `llm.output_messages` require an `LLM` span carrying
a full message structure. Spans of other kinds record images with the top-level `input.images` /
`output.images` attributes instead — a `TOOL` span running OCR, a `CHAIN` step holding a browser
screenshot, an image-generation call that has no chat messages at all. These parallel `input.value`
/ `output.value`: they are valid on **any** span kind and carry no role or message-structure
semantics.

### Attribute Pattern

`<input|output>.images.<imageIndex>.image.<attribute>`

Where:
- `<imageIndex>` is the zero-based index of the image
- `<attribute>` is `url` (required) or `mime_type` (optional)

A tool span that receives a page scan and returns an annotated version:

```json
{
  "openinference.span.kind": "TOOL",
  "input.images.0.image.url": "data:image/png;base64,iVBORw0KGgo...",
  "output.images.0.image.url": "https://example.com/annotated.png",
  "output.images.0.image.mime_type": "image/png"
}
```

Multiple images are indexed:

```
input.images.0.image.url = "https://example.com/page-1.png"
input.images.1.image.url = "https://example.com/page-2.png"
```

### Semantics

- `image.url` carries the same value semantics as `message_content.image.image.url` — a link to the
  image or its base64 encoding. That is an absolute URI (`https://`, `http://`, `s3://`, `gs://`, or
  a provider-specific scheme), a base64 data URI (`data:<mime>;base64,<payload>`), or a bare base64
  payload.
- Producers SHOULD emit a data URI in preference to a bare base64 payload when the MIME type is
  known, so that a consumer can identify the bytes without depending on `image.mime_type`.
- `image.mime_type` is OPTIONAL and SHOULD be set whenever `image.url` does not carry the type — a
  bare base64 payload, or a reference URI with no recoverable type such as `s3://` or `gs://`. It is
  redundant alongside a data URI and MAY be omitted there; where both are present and disagree, the
  data URI is authoritative.
- These attributes are additive, not a replacement for message content. An `LLM` span that already
  records an image under `message.contents` SHOULD NOT repeat it here.
- Order is carried by the index: producers SHOULD emit images in the order they occur, and
  consumers MAY rely on it. That is the only structure present — there is no role, no interleaving
  with text, and no correspondence to message indices; use `message.contents` when order relative
  to text matters.
- `input.value` / `output.value` remain the textual I/O of the span and are unaffected.

### Redaction and Size Limits

The controls that apply to message-content images apply to these attributes in the same way:

- `OPENINFERENCE_HIDE_INPUT_IMAGES` removes `input.images.*`, including the sibling
  `image.mime_type`.
- `OPENINFERENCE_HIDE_INPUTS` and `OPENINFERENCE_HIDE_OUTPUTS` remove `input.images.*` and
  `output.images.*` respectively, as they do for `input.value` / `output.value`.
- `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` and a configured blob uploader apply to
  `<input|output>.images.<i>.image.url` on the same terms as `message_content.image.image.url`
  (see [External Storage for Large Media](#external-storage-for-large-media)). Hiding takes
  precedence: a hidden image is never uploaded.

The size limit and externalization recognize base64 by the `data:` prefix, so a bare base64 payload
is measured by neither — the same as for message-content images. This is a further reason to prefer
a data URI when the MIME type is known.

Each control reaches these attributes wherever the SDK already implements it for message-content
images, and no further. The size limit is implemented in Python and JavaScript; blob
externalization is implemented in Python only, so on JavaScript an oversized payload is replaced
with `"__REDACTED__"` rather than uploaded. The Java and Go SDKs implement neither, for message
images or for these, so a payload budget configured there is not applied.


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
