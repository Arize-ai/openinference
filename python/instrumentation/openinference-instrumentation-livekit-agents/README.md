# OpenInference livekit-agents instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-livekit-agents.svg)](https://pypi.org/project/openinference-instrumentation-livekit-agents/)

This package maps livekit-agents voice turns onto OpenInference spans.

## Installation

```shell
pip install openinference-instrumentation-livekit-agents
```

## Usage

Register the instrumentor on the same provider that you pass to
`livekit.agents.telemetry.set_tracer_provider`:

```python
from livekit.agents.telemetry import set_tracer_provider
from openinference.instrumentation.livekit_agents import LiveKitAgentsInstrumentor
from opentelemetry.sdk.trace import TracerProvider

provider = TracerProvider()
LiveKitAgentsInstrumentor().instrument(tracer_provider=provider)
set_tracer_provider(provider)
```

One spoken question becomes a `conversation.turn` span of kind `AUDIO`, with user,
assistant, and tool child spans. The instrumentor uses the room SID as `session.id`
unless the caller already set a session ID. LiveKit stamps that room SID only inside
a worker job, so a session started directly on a room has no `session.id`.

The spans omit `input.audio.url` and `output.audio.url`. LiveKit spans carry transcripts
and token counts, not PCM audio.

LiveKit's own spans still export alongside the OpenInference turn. Each `assistant` span
repeats the token counts of the LiveKit `llm_request` it maps. A backend that prices
`llm_request` spans as LLM calls, as Arize does, counts each model call twice. To avoid that,
filter the raw `livekit-agents` `llm_request` spans out of the export.
