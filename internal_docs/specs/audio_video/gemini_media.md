# Gemini and Vertex media parts

How Google Gemini (and Vertex `GenerativeModel`) attach audio and video to `generateContent`. Image parts already map to `message_content.image`. Audio and video use the same `Part` union, discriminated by MIME type.

Permalinks use commit `473bf4b6b5a69e5324a5d4bac0fe852351338c43` of [googleapis/python-genai](https://github.com/googleapis/python-genai). Vertex AI Python uses `vertexai.generative_models.Part.from_uri` with the same `file_uri` plus `mime_type` pair.

---

## 1. Definitions

**FileData.** A URI plus MIME type. Typical schemes are `gs://`, `https://`, and Gemini File API URIs.

**Blob (`inline_data`).** Raw bytes plus MIME type. OpenInference stores these as a data URI.

**VideoMetadata.** Optional start offset, end offset, and fps on a video part. OpenInference does not publish these fields yet. No consumer walks them.

**MIME discriminator.** `image/*` is an image content item. `audio/*` is audio. `video/*` is video. Do not put video bytes in `image.url`.

---

## 2. Input params

[source](https://github.com/googleapis/python-genai/blob/473bf4b6b5a69e5324a5d4bac0fe852351338c43/google/genai/types.py)

```python
class FileData:
    file_uri: Optional[str]
    mime_type: Optional[str]

class Blob:
    data: Optional[bytes]
    mime_type: Optional[str]

class Part:
    text: Optional[str]
    file_data: Optional[FileData]
    inline_data: Optional[Blob]
    video_metadata: Optional[VideoMetadata]
```

URI video:

```python
from google.genai import types

types.Part.from_uri(
    file_uri="gs://cloud-samples-data/generative-ai/video/animals.mp4",
    mime_type="video/mp4",
)
```

Inline audio:

```python
types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
```

OpenInference rewrite (URI video):

```
llm.input_messages.0.message.contents.0.message_content.type = "video"
llm.input_messages.0.message.contents.0.message_content.video.video.url = "gs://cloud-samples-data/generative-ai/video/animals.mp4"
llm.input_messages.0.message.contents.0.message_content.video.video.mime_type = "video/mp4"
```

OpenInference rewrite (inline audio):

```
llm.input_messages.0.message.contents.0.message_content.type = "audio"
llm.input_messages.0.message.contents.0.message_content.audio.audio.url = "data:audio/wav;base64,..."
llm.input_messages.0.message.contents.0.message_content.audio.audio.mime_type = "audio/wav"
```

`file_uri` is copied verbatim into `video.url` or `audio.url`. Do not wrap `gs://` in a data URI.

---

## 3. Output

Gemini replies are usually text. If a later model returns `inline_data` or `file_data` with an audio or video MIME type, use `llm.output_messages` content items of type `"audio"` or `"video"` with the same nested leaves.

---

## 4. Instrumentor gap today

Python Vertex `_parse_part` and Google GenAI `_get_attributes_from_file_data` / `_get_attributes_from_inline_data` keep parts whose MIME type contains `"image"`. Video and audio parts are dropped from structured contents. They may still appear inside `input.value` JSON. The `future` demo span in `scripts/gemini_video_demo.py` is the target shape after those gates open.
