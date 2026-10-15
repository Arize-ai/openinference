# OpenInference LiveKit Agents Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-livekit-agents.svg)](https://pypi.org/project/openinference-instrumentation-livekit-agents/)

Python auto-instrumentation library for [LiveKit Agents](https://github.com/livekit/agents)
voice sessions. Each spoken turn becomes a trace with the user's and the agent's audio, the
transcripts, every model call, and every tool call.

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent
to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix)
or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Compatibility

| `openinference-instrumentation-livekit-agents` | `livekit-agents` | Python          |
| ---------------------------------------------- | ---------------- | --------------- |
| `>=0.1.0`                                      | `>=1.6.0`        | `>=3.10, <3.15` |

Turns are built from cascaded pipelines (speech-to-text, LLM, text-to-speech). Realtime
models (`RealtimeModel`) are not traced yet.

## Installation

```shell
pip install openinference-instrumentation-livekit-agents
```

## Quickstart

Register a tracer provider with `auto_instrument=True` in the agent server's `setup_fnc`.
LiveKit runs each job in its own process by default, and `setup_fnc` runs once in each one.

```python
from livekit.agents import AgentServer, JobProcess
from phoenix.otel import register  # or: from arize.otel import register


def setup(proc: JobProcess) -> None:
    register(project_name="livekit-voice-agent", auto_instrument=True)


server = AgentServer(setup_fnc=setup)
```

To instrument explicitly instead, pass a tracer provider:

```python
from openinference.instrumentation.livekit_agents import LiveKitAgentsInstrumentor

LiveKitAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
```

LiveKit's own telemetry does not need to be configured.

## What a turn looks like

Each reply the agent speaks becomes a `conversation.turn` span of kind `AUDIO`, emitted when
the reply ends:

- The turn holds `input.audio.url` (what the user said) and `output.audio.url` (what the agent
  played) as WAV data URIs, plus `input.value`, `output.value`, and `end_reason`
  (`complete` or `interrupted`).
- A `user` span holds the transcript and the speech-to-text model.
- An `assistant` span of kind `LLM` covers each model call, with input and output messages,
  token counts, and `time_to_first_token_ms`.
- Each function tool call is a `TOOL` span under the `assistant` span that requested it.

When the user interrupts, `output.audio.url` holds only the audio played before the
interruption, and `output.value` is LiveKit's record of the text that was played.

Every span carries the room SID as `session.id`, unless the caller set a session with
`using_session`. The SID is known inside a worker job, or when the session starts on a
connected `rtc.Room`.

## Audio

Audio is captured from the session's input and output audio streams. Each clip is a valid WAV,
trimmed by whole frames to `OPENINFERENCE_BASE64_AUDIO_MAX_LENGTH` characters. The default is
4,000,000, about 62 seconds of 24 kHz mono audio. Arize AX stores the data URIs as blobs, and
Phoenix keeps them inline.

To drop a direction, set `OPENINFERENCE_HIDE_INPUT_AUDIO` or `OPENINFERENCE_HIDE_OUTPUT_AUDIO`,
or use `TraceConfig(hide_inputs=True)` or `TraceConfig(hide_outputs=True)`. Audio for a hidden
direction is not buffered.

## LiveKit's own spans

livekit-agents emits its own spans (`agent_session`, `llm_request`, and others) to the global
tracer provider. They repeat the turn's model calls and token counts, which a backend would
price twice. By default the instrumentor points LiveKit's tracer at a no-op provider, and
`uninstrument()` restores it. To keep LiveKit's spans, instrument with `exclusive=False`.
Calling `livekit.agents.telemetry.set_tracer_provider` after instrumenting also brings them back.

## Examples

[examples/](examples/) has a voice agent with a weather tool, auto-instrumented and traced to
Arize AX (`arize_voice_agent.py`) or Phoenix (`phoenix_voice_agent.py`). `talk.py` joins the
room and asks the agent a question out loud, so the examples run without a microphone.

The examples need a LiveKit server and an OpenAI API key. For a local server, install
[`livekit-server`](https://docs.livekit.io/transport/self-hosting/local/) and run
`livekit-server --dev`, which accepts the key `devkey` and the secret `secret`.

```shell
pip install -r examples/requirements.txt
export OPENAI_API_KEY=...
export LIVEKIT_URL=ws://127.0.0.1:7880 LIVEKIT_API_KEY=devkey LIVEKIT_API_SECRET=secret

phoenix serve                                # in one terminal (pip install arize-phoenix)
python examples/phoenix_voice_agent.py dev   # in another
python examples/talk.py                      # in a third
```

For Arize AX, also set `ARIZE_SPACE_ID` and `ARIZE_API_KEY`, and run `arize_voice_agent.py`.
