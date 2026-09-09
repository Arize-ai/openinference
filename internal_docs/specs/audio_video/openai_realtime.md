# OpenAI Agents realtime audio

This document is the from-scratch mapping for `agents.realtime.RealtimeSession`, based on the shipping instrumentor in `_realtime.py`. Chat Completions `input_audio` is a different API and stays on `message.contents`. See [openai_chat_audio.md](./openai_chat_audio.md).

Today's capture lives in `python/instrumentation/openinference-instrumentation-openai-agents/src/openinference/instrumentation/openai_agents/_realtime.py`.

The wrapper is `make_realtime_wrapper`. It patches `RealtimeSession._put_event`. `make_send_audio_wrapper` patches `send_audio`. `make_close_wrapper` patches `close`. Tests live in `tests/test_realtime.py`.

---

## 1. What the instrumentor records today

Realtime is an event stream, not `messages[]`. The instrumentor keeps per-session state (`_RealtimeSessionState`) and one `_TurnState` at a time. Each turn owns zero or more `_UserInputState` objects and a dict of `_AssistantResponseState` objects keyed by `response_id`.

### Span tree

Opened in `_start_turn`, `on_speech_started` or `on_user_text_created`, `on_response_created`, and `on_function_call_added`:

```
AUDIO  "conversation.turn"     parent. Kind string AUDIO. Not in OpenInferenceSpanKindValues.
├─ USER  "user"                one per utterance or typed item. Kind string USER.
├─ LLM   "assistant"           one per response.id. Kind LLM.
│  └─ TOOL "<tool_name>"       one per function_call call_id. Kind TOOL.
└─ ...                         extra USER or LLM siblings for split input or tool follow-up
```

Turn boundaries are in `_turn_for_new_user_input`. A new user utterance after an assistant response finalizes the previous turn and starts another. A barge-in during a tool round-trip marks the originating turn `INTERRUPTED` and opens a new turn. The follow-up LLM span still attaches to the originating turn via `_turn_awaiting_followup`.

### Events that feed the buffers

`_dispatch_event` handles SDK wrappers. `_dispatch_raw` handles `RealtimeRawModelEvent` server dicts.

| Event | Handler | Effect |
|---|---|---|
| `RealtimeSession.send_audio` | `on_send_audio` | PCM into the active speech USER, or a 3 s pre-speech rolling buffer |
| `input_audio_buffer.speech_started` | `on_speech_started` | Open USER span. Flush pre-speech PCM into `user_audio_buf`. |
| `input_audio_buffer.append` | `on_user_audio_append` | More user PCM |
| `input_audio_buffer.committed` | `on_user_audio_committed` | Mark the utterance complete. Stamp TTFT origin. |
| `conversation.item.input_audio_transcription.completed` | `on_user_transcript_completed` | `user.user_transcript` |
| `conversation.item.added` / `.created` / `.done` with `input_text` | `on_user_text_created` | Open a `text_only` USER span. `input.value` at finalize. |
| `session.created` / `session.updated` | `on_session_created`, `on_session_config` | `session.id`, model, invocation-parameter snapshot |
| `response.created` | `on_response_created` | Open LLM span |
| `RealtimeAudio` | `on_audio_delta` | Assistant PCM into `asst_audio_buf`. First delta stamps TTFT. |
| `response.output_audio_transcript.delta` / `.done` | `on_asst_transcript_delta`, `on_asst_transcript_done` | Assistant transcript |
| `response.output_item.added` (`function_call`) | `on_function_call_added` | Open TOOL span under the LLM span |
| `response.function_call_arguments.done` | `on_function_call_arguments_done` | TOOL `input.value` |
| `conversation.item.*` (`function_call_output`) | `on_function_call_output` | TOOL `output.value`, then end the TOOL span |
| `response.done` | `on_response_done` | `_finalize_response`. Keep the turn open when `output` contains `function_call`. |
| `RealtimeAudioInterrupted` | `on_audio_interrupted` | Sticky `end_reason=interrupted` when a response is still in flight |
| `RealtimeError`, `RealtimeSession.close`, input-audio timeout | `on_error`, `on_session_close` | Finalize every open turn |

`RealtimeAgentEndEvent` is a no-op. Closing the latest turn on agent-end used to create empty AUDIO spans on barge-in.

### Where attributes are written

Media is not set when events arrive. Buffers accumulate, then `_finalize_user` and `_finalize_response` write the span. `_finalize_turn` then writes aggregated text on the AUDIO parent and ends every child that is still open.

| Writer | Span | Attributes today |
|---|---|---|
| `_finalize_user` | USER | `input.audio.url`, `input.audio.transcript` from PCM plus transcription. Typed turns set `input.value` and `input.mime_type` instead. |
| `_finalize_response` | LLM | `output.audio.url`, `output.audio.transcript`, token counts, `llm.model_name`, `llm.finish_reason`, `time_to_first_token_ms` |
| `_set_turn_io_attributes` | AUDIO | `input.value` joined from typed text and audio transcripts. `output.value` joined from assistant transcripts. No WAV on the parent. |
| `_start_turn` | AUDIO | `openinference.span.kind=AUDIO`, `llm.model_name`, `llm.invocation_parameters` from the session.update snapshot |

The four audio keys are instrumentor-local string literals (`_INPUT_AUDIO_URL` and friends). Comments in `_realtime.py` still say they are not yet in semconv. After this spec, compose them from `AudioAttributes` instead of repeating the strings.

### PCM and transcripts

OpenAI Realtime streams 24 kHz mono PCM16. `pcm16_to_wav_data_uri` wraps the buffer as `data:audio/wav;base64,...`. User PCM comes from `send_audio` (pre-speech rolling buffer, then the active USER) and from `input_audio_buffer.append`. Assistant PCM comes from `RealtimeAudio` deltas (`on_audio_delta`). User transcripts come from `conversation.item.input_audio_transcription.completed`. Assistant transcripts come from `response.output_audio_transcript.done`. OpenAI also emits that `.done` event when a response is interrupted, incomplete, or cancelled. Prefer the terminal `.done` transcript. Join `.delta` chunks only when `.done` is missing. `output.audio.transcript` is the generated transcript. It is not what the user heard. Cancellation does not say where playback stopped. Playback position is a client concern (`conversation.item.truncate` with `audio_end_ms`).

Typed user text is a separate USER span (`text_only=True`). It must not receive mic PCM or audio transcripts. That split is `on_user_text_created` versus `on_speech_started`.

### Hide and truncation (today)

`_hide_input_audio` is `TraceConfig.hide_inputs` or `OPENINFERENCE_HIDE_INPUT_AUDIO`. `_hide_output_audio` is `TraceConfig.hide_outputs` or `OPENINFERENCE_HIDE_OUTPUT_AUDIO`. Today the instrumentor slices audio data URIs with `truncate_audio_data_uri` and `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` (default 32000). The published size gate is externalize or redact, not slice. Audio-specific `TraceConfig` fields are a later promotion. The env vars already match [Configuration](../../../spec/configuration.md).

---

## 2. Convention if this were designed today

The published convention has two attachment points and one leaf vocabulary. See [vendor_comparison.md](./vendor_comparison.md). Chat content parts use `message.contents`. Voice sessions that are not a chat `messages[]` list use span-root `input.audio.*` and `output.audio.*`.

Realtime is the second case. Designing it now, with that rule in hand, yields the same span tree and the same four keys. The change is how the keys are spelled in code, not a new tree.

**Keep the AUDIO, USER, LLM, TOOL tree.** `speech_started`, `response.created`, and `function_call` are independent lifetimes. USER spans record one utterance, so the WAV belongs on span-root `input.audio.*`. `llm.input_messages` is for LLM spans that wrap a chat `messages[]` list. Merging USER into the LLM span would hide barge-in and split utterances.

**Keep WAV on the child that owns the buffer.** USER owns `user_audio_buf`. LLM owns `asst_audio_buf`. The AUDIO parent stays a turn summary (`input.value` and `output.value` transcripts, session config, `end_reason`). Do not copy the data URI onto the parent.

**Compose leaves from `AudioAttributes`.** `AudioAttributes.AUDIO_URL` is `audio.url`. Prefix `input.` or `output.`. Do not keep parallel string literals.

```
input.audio.url        = "input." + AudioAttributes.AUDIO_URL
input.audio.transcript = "input." + AudioAttributes.AUDIO_TRANSCRIPT
output.audio.url        = "output." + AudioAttributes.AUDIO_URL
output.audio.transcript = "output." + AudioAttributes.AUDIO_TRANSCRIPT
```

**USER spans stay span-root.** Optional later work can add `llm.output_messages` audio contents on the LLM span so Phoenix's message walker can render assistant audio without a second code path. That is additive. It does not replace span-root keys on USER.

**Leave AUDIO and USER kinds unpublished.** One producer is not a taxonomy. TOOL and LLM already use `OpenInferenceSpanKindValues`.

**Promote hide flags onto `TraceConfig`.** The env vars stay as the spec names. `_realtime.py` should read the same fields every other instrumentor will use, once those fields exist.

---

## 3. Event to attribute mapping

Each row is one capture the instrumentor already implements. The published key is the same string. The "today" column is the local constant. The "designed today" column is the composition that should replace it.

### User audio (USER span, `_finalize_user`)

| Source in `_realtime.py` | Buffer or field | Today | Designed today |
|---|---|---|---|
| `on_send_audio`, `on_user_audio_append` | `user.user_audio_buf` then `pcm16_to_wav_data_uri` | `_INPUT_AUDIO_URL` | `"input." + AudioAttributes.AUDIO_URL` |
| `on_user_transcript_completed` (`conversation.item.input_audio_transcription.completed`) | `user.user_transcript` | `_INPUT_AUDIO_TRANSCRIPT` | `"input." + AudioAttributes.AUDIO_TRANSCRIPT` |
| `on_user_text_created` (`input_text` or `text` parts) | `user.user_text` | `SpanAttributes.INPUT_VALUE` plus `INPUT_MIME_TYPE` `text/plain` | unchanged |

Do not write audio keys on `text_only` USER spans. Tests already assert that (`test_realtime.py`).

### Assistant audio (LLM span, `_finalize_response`)

| Source in `_realtime.py` | Buffer or field | Today | Designed today |
|---|---|---|---|
| `on_audio_delta` (`RealtimeAudio`) | `response.asst_audio_buf` then WAV URI | `_OUTPUT_AUDIO_URL` | `"output." + AudioAttributes.AUDIO_URL` |
| `on_asst_transcript_done` (`response.output_audio_transcript.done`, including interrupted, incomplete, or cancelled responses). Join `asst_transcript_deltas` only when `.done` is missing. | `response.asst_transcript` (generated transcript, not playback position) | `_OUTPUT_AUDIO_TRANSCRIPT` | `"output." + AudioAttributes.AUDIO_TRANSCRIPT` |
| `on_response_done` usage | `input_token_details.audio_tokens` | `LLM_TOKEN_COUNT_PROMPT_DETAILS_AUDIO` | unchanged |
| `on_response_done` usage | `output_token_details.audio_tokens` | `LLM_TOKEN_COUNT_COMPLETION_DETAILS_AUDIO` | unchanged |

### Turn parent (AUDIO span)

No `audio.url` on this span. Transcripts already flow into `input.value` and `output.value` with the same hide split as the children (`_set_turn_io_attributes`). Keep that.

---

## 4. How to update `_realtime.py`

This is the remaining constant-composition follow-up. This PR already dropped `input.audio.mime_type` and `output.audio.mime_type`.

1. Import `AudioAttributes` next to `SpanAttributes`.
2. Replace the four string literals with compositions. Keep the same local names so `_finalize_user` and `_finalize_response` stay unchanged:

```python
_INPUT_AUDIO_URL = f"input.{AudioAttributes.AUDIO_URL}"
_INPUT_AUDIO_TRANSCRIPT = f"input.{AudioAttributes.AUDIO_TRANSCRIPT}"
_OUTPUT_AUDIO_URL = f"output.{AudioAttributes.AUDIO_URL}"
_OUTPUT_AUDIO_TRANSCRIPT = f"output.{AudioAttributes.AUDIO_TRANSCRIPT}"
```

3. Replace the comment that cites `spec/audio_spans.md` (that file does not exist) with a pointer to this document and to [Multimodal Attributes](../../../spec/multimodal_attributes.md#span-root-audio).
4. Keep `_AUDIO_KIND` and `_USER_KIND` as local strings. Do not add them to `OpenInferenceSpanKindValues` in the same change.
5. When `TraceConfig` grows `hide_input_audio` and `hide_output_audio`, point `_hide_input_audio` and `_hide_output_audio` at those fields and keep the env vars as the fallback, matching other hide flags. Same for `base64_audio_max_length`.
6. Leave `llm.input_messages` and `message_content.audio` off this change. Tests in `test_realtime.py` assert the flat url and transcript keys and the USER versus LLM split. Keep asserting that mime keys are absent.

### What not to change in that follow-up

- Event dispatch (`_dispatch_event`, `_dispatch_raw`)
- Turn and barge-in lifecycle
- WAV encoding (`pcm16_to_wav_data_uri`)
- TOOL spans keyed by `call_id`
- Session context (`session.id` from `session.created`)
- Dual-write of assistant audio onto `llm.output_messages` (separate, optional)

---

## 5. Hide and size gates

These names are now in [Configuration](../../../spec/configuration.md). openai-agents already honors them in `_realtime.py`. Other voice instrumentors should match.

| Variable | Effect in this instrumentor |
|---|---|
| `OPENINFERENCE_HIDE_INPUT_AUDIO` | Skip `input.audio.url` and `input.audio.transcript` on USER. Also skip audio-derived `input.value` on the AUDIO parent. |
| `OPENINFERENCE_HIDE_OUTPUT_AUDIO` | Skip `output.audio.url` and `output.audio.transcript` on LLM. Also skip `output.value` on the AUDIO parent. |
| `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` | Size gate for audio data URIs. Externalize or redact over-limit payloads. Do not slice. Default `32000`. |

`TraceConfig(hide_inputs=True)` and `TraceConfig(hide_outputs=True)` already cascade.
