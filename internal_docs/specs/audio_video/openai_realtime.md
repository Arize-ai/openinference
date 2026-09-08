# OpenAI Agents realtime audio

How the openai-agents Python instrumentor records voice turns today, and which of those keys this spec publishes.

The shipping code is `python/instrumentation/openinference-instrumentation-openai-agents/src/openinference/instrumentation/openai_agents/_realtime.py`. The README documents the span tree.

---

## 1. Definitions

**Realtime session.** `agents.realtime.RealtimeSession`. PCM16 buffers and transcripts arrive as SDK events, not as chat `messages[]`.

**USER span.** One span per user turn. Kind string `USER`. Not in `OpenInferenceSpanKindValues`. This spec does not add it.

**AUDIO span.** Parent span named `conversation.turn`. Kind string `AUDIO`. Not in `OpenInferenceSpanKindValues`. This spec does not add it.

**Span-root audio.** `input.audio.url` (and mime type, transcript) on the USER span, and `output.audio.*` on the LLM span. These strings already ship. This spec publishes the leaves.

---

## 2. Input (user audio)

The instrumentor wraps PCM16 as a WAV data URI.

```
input.audio.url = "data:audio/wav;base64,..."
input.audio.mime_type = "audio/wav"
input.audio.transcript = "What's the weather in Tokyo?"
```

Composition is the `input` prefix plus existing `AudioAttributes` leaves (`audio.url`, `audio.mime_type`, `audio.transcript`). Typed text on the same turn still uses `input.value`.

Do not also require `llm.input_messages` on the USER span. That span is not a chat completion.

---

## 3. Output (assistant audio)

On the LLM span for the assistant turn:

```
output.audio.url = "data:audio/wav;base64,..."
output.audio.mime_type = "audio/wav"
output.audio.transcript = "It's sunny and 24 C in Tokyo."
```

Token counts already use `llm.token_count.prompt_details.audio` and `llm.token_count.completion_details.audio`. Those stay as they are.

---

## 4. Hide and size gates

These environment variables already work in `_realtime.py`. This spec lists them so other voice instrumentors can match.

| Variable | Effect |
|---|---|
| `OPENINFERENCE_HIDE_INPUT_AUDIO` | Drop `input.audio.url`, `input.audio.mime_type`, and `input.audio.transcript` |
| `OPENINFERENCE_HIDE_OUTPUT_AUDIO` | Drop `output.audio.url`, `output.audio.mime_type`, and `output.audio.transcript` |
| `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` | Truncate the base64 body of audio data URIs. Keep the `data:audio/wav;base64,` prefix. Default `32000`. |

`TraceConfig(hide_inputs=True)` and `TraceConfig(hide_outputs=True)` already cascade. Promoting the audio-specific flags onto shared `TraceConfig` fields is a follow-up.

---

## 5. What this spec does not change

The shipped USER and AUDIO kind strings stay instrumentor-local. A later span-kind RFC can publish them. Chat Completions `input_audio` parts stay on `message.contents`. See [openai_chat_audio.md](./openai_chat_audio.md).
