# OpenInference Pipecat Instrumentation

Python auto-instrumentation library for Pipecat. This library allows you to convert Pipecat traces to OpenInference, which is OpenTelemetry compatible, and view those traces in [Arize Phoenix](https://github.com/Arize-ai/phoenix).

## Compatibility

| `openinference-instrumentation-pipecat` | `pipecat-ai`     | Python   |
| --------------------------------------- | ---------------- | -------- |
| `>=1.0`                                 | `>=1.0`          | `>=3.11` |
| `<=0.1.4`                               | `<1.0` (e.g. `0.0.99`) | `>=3.10` |

Pipecat 1.0 introduced breaking changes (renamed observers, removed
`LLMMessagesFrame`, dropped Python 3.10). If you're still on `pipecat-ai<1.0`,
pin this instrumentor to `<=0.1.4`:

```shell
pip install 'openinference-instrumentation-pipecat<=0.1.4' 'pipecat-ai<1.0'
```

> **Breaking change: turn model.** Beginning with this release, bot-first
> sessions no longer produce a separate "greeting" turn. The bot greeting and
> the user's first reply now form a single round-trip turn (turn 1).
> Dashboards or downstream code that counted turns per session will see one
> fewer turn per session in bot-first apps. Span schema is otherwise
> unchanged.

## Installation

```shell
pip install openinference-instrumentation-pipecat
```

## Quickstart

This quickstart shows you how to setup tracing in your Pipecat application:

```python
from phoenix.otel import register
from openinference.instrumentation.pipecat import PipecatInstrumentor

# Set up the tracer provider
tracer_provider = register(
    project_name="default" #Phoenix project name
)

# Add auto-instrumentor at the top of the application
PipecatInstrumentor().instrument(tracer_provider=tracer_provider)

# Build your Pipecat pipeline
# ... (add your Pipecat pipeline code here)

### CONFIGURATION SETUP (Transport, LLM, STT, TTS) ###

### PIPELINE ###
pipeline = Pipeline(...)

### TASK ###
task = PipelineTask(
    pipeline,
    conversation_id=conversation_id,  # conversation id is used for session tracking in Arize or Phoenix
)

### EVENT HANDLING
@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    await task.queue_frames([LLMRunFrame()])

### PIPELINE RUNNER ###
runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
await runner.run(task)
```

After configuring tracing, exchanges in the running application are logged to your project in Phoenix or Arize AX.


## Conversation turn model

Each `pipecat.conversation.turn` span represents a single round-trip exchange between a user and the bot. Either party can initiate the turn: a user-initiated turn begins when the user starts speaking, while a bot-initiated turn begins when the bot speaks first. In both directions, the turn closes once the responder finishes — or once the configured timeouts elapse.

The following turn span attributes are relevant to the bidirectional model:

- `conversation.initiator` (new): `"user"` or `"bot"` — identifies which party started the exchange.
- `conversation.end_reason`: now one of `"completed"`, `"interrupted"`, `"no_responder_timeout"`.

Two timeouts govern when a turn closes. The responder timeout is configurable on `PipelineTask`:

```python
task = PipelineTask(
    pipeline,
    conversation_id=conversation_id,
    _no_responder_timeout_secs=15.0,  # default 10 — wait this long for the responder
)
```

The existing `turn_end_timeout_secs` constructor kwarg on `OpenInferenceObserver` (default `2.5`) still governs the inactivity gap after a completed exchange.


## Example

1. Install dependencies
```bash
uv pip install -e '.[examples]'
```

or

```bash
uv pip install -r examples/trace/requirements.txt
```

2. Run example
```bash
uv run python examples/trace/001-trace.py
```
