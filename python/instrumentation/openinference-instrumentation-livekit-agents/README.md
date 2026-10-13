# OpenInference livekit-agents instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-livekit-agents.svg)](https://pypi.org/project/openinference-instrumentation-livekit-agents/)

This package traces [livekit-agents](https://github.com/livekit/agents) voice sessions with
OpenInference spans, including the audio of each turn.

## Installation

```shell
pip install openinference-instrumentation-livekit-agents
```

## Usage

Instrument before the session starts, in the process that runs the job:

```python
from openinference.instrumentation.livekit_agents import LiveKitAgentsInstrumentor

LiveKitAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
```

LiveKit's own telemetry does not need to be configured. Under the default process job
executor, instrument from the server's `setup_fnc` so each job process is instrumented.
[examples/voice_agent.py](examples/voice_agent.py) shows the pattern.

## What a turn looks like

Each reply the agent speaks becomes a `conversation.turn` span of kind `AUDIO`, rendered when
the reply ends:

- The turn holds `input.audio.url` (what the user said) and `output.audio.url` (what the agent
  played) as WAV data URIs, plus `input.value`, `output.value`, and `end_reason`.
- A `user` span holds the transcript and the speech-to-text model.
- An `assistant` span of kind `LLM` covers each model call, with input and output messages,
  token counts, and `time_to_first_token_ms`.
- Each function tool call is a `TOOL` span under the `assistant` span that requested it.

After a barge-in, `output.value` and `output.audio.url` hold only what the user heard. The
room SID is `session.id` unless the caller set a session with `using_session`. LiveKit knows
the SID inside a worker job or when the session starts on a connected room.

## Audio

Audio is captured from the session's input and output audio streams. Each clip is a valid WAV
trimmed by whole frames to `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` characters, 4,000,000 by
default (about 62 seconds of 24 kHz mono audio). Set `OPENINFERENCE_HIDE_INPUT_AUDIO` or
`OPENINFERENCE_HIDE_OUTPUT_AUDIO` to drop a direction. `TraceConfig(hide_inputs=True)` and
`TraceConfig(hide_outputs=True)` drop it too. Arize stores the data URIs as blobs. Phoenix
keeps them inline.

## Double counting

livekit-agents exports its own spans to the global tracer provider. When that provider sends
to the same project, a backend that prices LiveKit's `llm_request` spans counts each model call
twice. Pass `set_global_tracer_provider=False` to `register`, as the examples do, to keep
LiveKit's spans out.

## Examples

[examples/](examples/) runs a voice agent with a weather tool against a LiveKit server, traced
to Arize (`arize_voice_agent.py`) or Phoenix (`phoenix_voice_agent.py`). `talk.py` joins the
room and asks a question out loud, so the examples run without a person:

```shell
pip install -r examples/requirements.txt
python examples/arize_voice_agent.py dev   # or phoenix_voice_agent.py
python examples/talk.py                    # in a second terminal
```

Realtime models (`RealtimeModel`) are not traced yet.
