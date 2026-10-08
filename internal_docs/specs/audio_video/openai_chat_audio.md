# OpenAI Chat Completions audio

How OpenAI attaches audio to `POST /v1/chat/completions`. Video is not a content part on this API. Image parts stay on `image_url` and map to existing `message_content.image` keys.

Permalinks use tag `v1.52.2` of [openai/openai-python](https://github.com/openai/openai-python). Behavior is checked against the [audio guide](https://developers.openai.com/api/docs/guides/audio).

---

## 1. Definitions

**Input audio part.** A user message content item with `type: "input_audio"`. The payload is raw base64 plus a format literal (`wav` or `mp3`). It is not a URL.

**Assistant audio object.** When the request asks for spoken output (`modalities` includes `"audio"`), the assistant message carries a sibling `audio` object with `id`, base64 `data`, `transcript`, and `expires_at`. That object is not a content part.

**Format.** OpenAI's `wav` or `mp3` literal. Map it to MIME `audio/wav` or `audio/mpeg` when writing OpenInference.

---

## 2. Input params

[source](https://github.com/openai/openai-python/blob/v1.52.2/src/openai/types/chat/chat_completion_content_part_input_audio_param.py)

```python
class InputAudio(TypedDict):
    data: str                          # base64, no data: prefix
    format: Literal["wav", "mp3"]

class ChatCompletionContentPartInputAudioParam(TypedDict):
    type: Literal["input_audio"]
    input_audio: InputAudio
```

Example request fragment:

```json
{
  "model": "gpt-4o-audio-preview",
  "modalities": ["text", "audio"],
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this recording?"},
        {
          "type": "input_audio",
          "input_audio": {
            "data": "<base64>",
            "format": "wav"
          }
        }
      ]
    }
  ]
}
```

OpenInference rewrite of the input audio part:

1. Build a data URI. `data:audio/wav;base64,<data>` when `format` is `wav`. `data:audio/mpeg;base64,<data>` when `format` is `mp3`.
2. Emit a `message.contents` item with `message_content.type = "audio"`.
3. Set `message_content.audio.audio.url` to that data URI.
Do not copy `format` as its own attribute and do not emit `audio.mime_type`. The data URI prefix already carries the MIME type.

---

## 3. Output

Assistant audio lives on `message.audio`, not on `content[]`.

```json
{
  "role": "assistant",
  "content": null,
  "audio": {
    "id": "audio_abc",
    "data": "<base64 wav>",
    "transcript": "The recording is a weather forecast.",
    "expires_at": 1735689600
  }
}
```

OpenInference rewrite on `llm.output_messages`:

```
llm.output_messages.0.message.role = "assistant"
llm.output_messages.0.message.contents.0.message_content.type = "audio"
llm.output_messages.0.message.contents.0.message_content.id = "audio_abc"
llm.output_messages.0.message.contents.0.message_content.audio.audio.url = "data:audio/wav;base64,<data>"
llm.output_messages.0.message.contents.0.message_content.audio.audio.transcript = "The recording is a weather forecast."
```

Use `output_messages` because this is still a chat completion message. Do not put it on a voice-session span. openai-agents realtime uses instrumentor-local `output.audio.*` for that case.

---

## 4. Responses API

The OpenAI Responses API also accepts `input_audio` content parts. The Python OpenAI instrumentor still has a TODO for them. Map those parts the same way as Chat Completions once the instrumentor captures them. File ids stay a later `file` content type, not `audio.url`.
