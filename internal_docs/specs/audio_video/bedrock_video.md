# Amazon Bedrock Converse video

How Bedrock Converse sends video on a user message. Images already map to `message_content.image`. Audio on Converse is not a first-class block in the same way. Document audio if a model adds it later using the chat `message.contents` audio keys.

AWS reference: [VideoBlock](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_VideoBlock.html), [VideoSource](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_VideoSource.html).

---

## 1. Definitions

**VideoBlock.** A Converse content block with a required `format` and a `source` union.

**Format.** One of `mkv`, `mov`, `mp4`, `webm`, `flv`, `mpeg`, `mpg`, `wmv`, or `three_gp`. Use `format` to build the data URI prefix for a bytes source (`mp4` becomes `data:video/mp4;base64,...`). Do not emit `video.mime_type`. Consumers infer MIME type from the URL path extension or from that data URI prefix.

**VideoSource.** Either `bytes` (base64, under 25 MB encoded) or `s3Location` (up to 1 GB).

---

## 2. Input params

```json
{
  "role": "user",
  "content": [
    {"text": "Summarize this clip."},
    {
      "video": {
        "format": "mp4",
        "source": {
          "s3Location": {
            "uri": "s3://bucket/clip.mp4"
          }
        }
      }
    }
  ]
}
```

S3 source rewrite:

```
llm.input_messages.0.message.contents.0.message_content.type = "text"
llm.input_messages.0.message.contents.0.message_content.text = "Summarize this clip."
llm.input_messages.0.message.contents.1.message_content.type = "video"
llm.input_messages.0.message.contents.1.message_content.video.video.url = "s3://bucket/clip.mp4"
```

Bytes source rewrite. Build `data:video/mp4;base64,<bytes>` when `format` is `mp4` (or the prefix that matches `format`) and store that string in `video.video.url`.

Do not add `video.format` as a published attribute. Bedrock JS already emits a non-spec `message_content.image.format` for images. Do not copy that for video.

---

## 3. Instrumentor gap today

Python Bedrock has `pass  # TODO: handle video tool result`. JS Bedrock Nova comments that it ignores video content. The mapping above is the target once those instrumentors emit video blocks.
