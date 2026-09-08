# Audio and video surfaces, vendor comparison

Side-by-side comparison of how OpenAI, Google Gemini, and Amazon Bedrock attach audio and video to model calls. Two axes:

1. **Chat multimodal.** Content parts on a chat or generateContent request.
2. **Voice session.** PCM buffers on a realtime or live session that is not a chat `messages[]` list.

Sibling docs:

- [openai_chat_audio.md](./openai_chat_audio.md)
- [openai_realtime.md](./openai_realtime.md)
- [gemini_media.md](./gemini_media.md)
- [bedrock_video.md](./bedrock_video.md)

Anthropic Messages accepts images and PDFs. It has no native audio or video content part. There is nothing to map until that changes.

---

## 0. At a glance

| Axis | OpenAI Chat Completions | OpenAI Agents realtime | Gemini / Vertex | Bedrock Converse |
|---|---|---|---|---|
| Audio in | `content[].type == "input_audio"` | PCM16 events on USER span | `Part` with `audio/*` MIME | not a first-class block |
| Audio out | `message.audio` object | PCM16 on LLM span | rare `inline_data` | not a first-class block |
| Video in | none (file ids later) | none | `Part` with `video/*` MIME | `content[].video` |
| Wire URL | none. Base64 plus `format` | WAV data URI built by the instrumentor | `file_uri` or inline bytes | `s3Location.uri` or bytes |
| Transcript | assistant `audio.transcript` | SDK transcription events | none on the part | none on the block |

---

## 1. Chat multimodal (side by side)

### Shape of a single media part

| Vendor | Container | Discriminator | Bytes or URI | Extra fields OpenInference ignores |
|---|---|---|---|---|
| OpenAI Chat | `messages[].content[]` | `type == "input_audio"` | base64 `data` plus `format` | none |
| Gemini | `contents[].parts[]` | MIME on `file_data` or `inline_data` | `file_uri` or `Blob.data` | `videoMetadata` |
| Bedrock | `messages[].content[]` | presence of `video` | `source.s3Location` or `source.bytes` | `format` after MIME mapping |

### Canonical JSON

OpenAI input audio:

```json
{
  "type": "input_audio",
  "input_audio": {"data": "<base64>", "format": "wav"}
}
```

Gemini video URI:

```json
{
  "fileData": {
    "fileUri": "gs://bucket/clip.mp4",
    "mimeType": "video/mp4"
  }
}
```

Bedrock video S3:

```json
{
  "video": {
    "format": "mp4",
    "source": {"s3Location": {"uri": "s3://bucket/clip.mp4"}}
  }
}
```

### URL construction rule

| Vendor payload | OpenInference `*.url` value |
|---|---|
| OpenAI `data` + `format` | `data:audio/<mime>;base64,<data>` |
| Gemini `file_uri` | the URI as sent (`gs://`, `https://`, File API) |
| Gemini `inline_data` | `data:<mime_type>;base64,<bytes>` |
| Bedrock `s3Location.uri` | that S3 URI |
| Bedrock `bytes` | `data:video/<mime>;base64,<bytes>` |

Never store a provider file id in `audio.url` or `video.url`. File ids wait for a `file` content type.

---

## 2. Voice session (OpenAI Agents only in this repo)

Realtime turns are not chat messages. The shipped tree is AUDIO (parent), USER, LLM, and TOOL. USER and AUDIO kinds stay unpublished.

Published keys on those spans:

| Span | Keys |
|---|---|
| USER | `input.audio.url`, `input.audio.mime_type`, `input.audio.transcript` |
| LLM | `output.audio.url`, `output.audio.mime_type`, `output.audio.transcript` |

Leaves are the same `AudioAttributes` used under `message.contents`. Prefixes differ because the span is not an LLM message list.

Keep this tree off `llm.input_messages`. Phoenix's chat renderer is not a reason to rewrite USER spans as fake chat completions. The event-level map and the `_realtime.py` constant swap are in [openai_realtime.md](./openai_realtime.md).

---

## 3. OpenInference attribute mapping

### Chat content items

Prefix every row with `llm.<input|output>_messages.<i>.message.contents.<j>.`.

| Provider field | OpenInference attribute | Notes |
|---|---|---|
| (discriminator) | `message_content.type` = `"audio"` or `"video"` | `"audio"` is already listed. This spec adds `"video"`. |
| OpenAI `input_audio.data` + `format` | `message_content.audio.audio.url` and `.mime_type` | Data URI. MIME from `format`. |
| OpenAI assistant `audio.data` | `message_content.audio.audio.url` on **output** messages | Still a chat message, not span-root `output.audio.*`. |
| OpenAI assistant `audio.transcript` | `message_content.audio.audio.transcript` | |
| OpenAI assistant `audio.id` | `message_content.id` | Replay or expiry lookup. |
| Gemini `file_uri` with `audio/*` | `message_content.audio.audio.url` | Copy URI. |
| Gemini `file_uri` with `video/*` | `message_content.video.video.url` | Copy URI. |
| Gemini `mime_type` | `audio.mime_type` or `video.mime_type` | SHOULD when the provider sends it. |
| Gemini `inline_data` | data URI in the matching `*.url` | |
| Bedrock `s3Location.uri` or `bytes` | `message_content.video.video.url` | |

Double nesting is intentional. It matches `message_content.image.image.url`. Constants compose as `MESSAGE_CONTENT_VIDEO` + `VIDEO_URL`.

### Span-root voice

These keys are singular on each span. One USER span owns one `user_audio_buf`. One LLM span owns one `asst_audio_buf`. Split speech or typed text is another USER sibling, not `input.audio.0.url`. Multipart arrays stay on `message.contents` for chat APIs.

| Provider field | OpenInference attribute |
|---|---|
| User PCM wrapped as WAV | `input.audio.url` |
| User MIME | `input.audio.mime_type` |
| User transcript | `input.audio.transcript` |
| Assistant PCM wrapped as WAV | `output.audio.url` |
| Assistant MIME | `output.audio.mime_type` |
| Assistant transcript | `output.audio.transcript` |

### GenAI dual-write (`message.contents` only)

Generalize `_image_part_from_url` to a MIME-aware helper. `GenAIModalityValues.VIDEO` already exists and is unused.

| OpenInference URL | GenAI part |
|---|---|
| `data:audio/…;base64,…` | `{type: "blob", modality: "audio", mime_type, content}` |
| `data:video/…;base64,…` | `{type: "blob", modality: "video", mime_type, content}` |
| `https://`, `gs://`, `s3://`, other absolute URI | `{type: "uri", modality: "audio" or "video", mime_type?, uri}` |

`audio.transcript` has no GenAI part. Span-root `input.audio.*` is not dual-written until a separate mapper exists.

Do not emit unmerged `gen_ai.prompt.{n}.content.{m}.media_uri`.

### Hide and offload

Message-content masking matches images. Hide or offload when the key contains `message_content.audio` or `message_content.video` and ends with `audio.url` or `video.url`.

Span-root masking matches openai-agents today. Hide `input.audio.*` and `output.audio.*` via `OPENINFERENCE_HIDE_INPUT_AUDIO` and `OPENINFERENCE_HIDE_OUTPUT_AUDIO`.

Blob upload already classifies `video/` MIME. Wire `TraceConfig.mask()` after constants land. Audio and video offload follow the image length-gate pattern with their own env vars.

---

## 4. Constants to add

Already shipped:

- `AudioAttributes.AUDIO_URL` = `audio.url`
- `AudioAttributes.AUDIO_MIME_TYPE` = `audio.mime_type`
- `AudioAttributes.AUDIO_TRANSCRIPT` = `audio.transcript`

New:

- `MessageContentAttributes.MESSAGE_CONTENT_AUDIO` = `message_content.audio`
- `MessageContentAttributes.MESSAGE_CONTENT_VIDEO` = `message_content.video`
- `VideoAttributes.VIDEO_URL` = `video.url`
- `VideoAttributes.VIDEO_MIME_TYPE` = `video.mime_type`

`message_content.type` allowed values add `"video"`.

Demo scripts keep these as string literals until every language package is bumped. See [scripts/README.md](./scripts/README.md).
