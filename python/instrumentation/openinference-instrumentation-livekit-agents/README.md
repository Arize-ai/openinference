# OpenInference livekit-agents instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-livekit-agents.svg)](https://pypi.org/project/openinference-instrumentation-livekit-agents/)

This package traces [livekit-agents](https://github.com/livekit/agents) voice sessions with
OpenInference spans, including the audio of each turn.

## Installation

```shell
pip install openinference-instrumentation-livekit-agents
```

## Usage

Register a tracer provider with `auto_instrument=True`, or call the instrumentor yourself,
before the session starts:

```python
from phoenix.otel import register  # or: from arize.otel import register

register(project_name="livekit-voice-agent", auto_instrument=True)
```

```python
from openinference.instrumentation.livekit_agents import LiveKitAgentsInstrumentor

LiveKitAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
```

LiveKit's own telemetry does not need to be configured. Under the default process job
executor, register from the server's `setup_fnc`, which runs in each job process, as the
[examples](examples/) do.

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

## LiveKit's own spans

livekit-agents emits its own spans (`agent_session`, `llm_request`, and others). By default
the instrumentor points LiveKit's tracer at a no-op provider, so those spans do not repeat the
turn's model calls and token counts, which a backend would otherwise price twice. To keep them,
instrument with `exclusive=False`. `uninstrument()` restores LiveKit's tracer.

## Examples

[examples/](examples/) has a voice agent with a weather tool, auto-instrumented and traced to
Arize (`arize_voice_agent.py`) or Phoenix (`phoenix_voice_agent.py`). `talk.py` joins the
room and asks a question out loud, so the examples run without a person:

```shell
pip install -r examples/requirements.txt
python examples/arize_voice_agent.py dev   # or phoenix_voice_agent.py
python examples/talk.py                    # in a second terminal
```

Realtime models (`RealtimeModel`) are not traced yet.
