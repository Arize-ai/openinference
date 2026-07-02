# OpenInference Pipecat Instrumentation

Python auto-instrumentation library for Pipecat. This library allows you to convert Pipecat traces to OpenInference, which is OpenTelemetry compatible, and view those traces in [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/).

## Compatibility

| `openinference-instrumentation-pipecat` | `pipecat-ai`           | Python   |
|-----------------------------------------| ---------------------- | -------- |
| `>=2.0`                                 | `>=1.3`                | `>=3.11` |
| `>=1.0, <2.0`                           | `>=1.0, <1.3`          | `>=3.11` |
| `<=0.1.4`                               | `<1.0` (e.g. `0.0.99`) | `>=3.10` |

Pipecat 1.3 deprecated `PipelineTask` in favor of `PipelineWorker`. Instrumentor
versions `<2.0` only wrap `PipelineTask`, so bots built on `PipelineWorker` get
no observer injection and emit no spans. Use `>=2.0` with `pipecat-ai>=1.3`.

Pipecat 1.0 introduced breaking changes (renamed observers, removed
`LLMMessagesFrame`, dropped Python 3.10). If you're still on `pipecat-ai<1.0`,
pin this instrumentor to `<=0.1.4`:

```shell
pip install 'openinference-instrumentation-pipecat<=0.1.4' 'pipecat-ai<1.0'
```

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

### WORKER ###
worker = PipelineWorker(
    pipeline,
    conversation_id=conversation_id,  # conversation id is used for session tracking in Arize or Phoenix
)

### EVENT HANDLING
@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    await worker.queue_frames([LLMRunFrame()])

### PIPELINE RUNNER ###
runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)
await runner.run(worker)
```

After configuring tracing, exchanges in the running application are logged to your project in [Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/).

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

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [More info on OpenInference and Arize AX](https://arize.com/)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
