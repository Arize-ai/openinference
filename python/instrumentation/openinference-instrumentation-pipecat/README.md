# OpenInference Pipecat Instrumentation

Python auto-instrumentation library for Pipecat. This library allows you to convert Pipecat traces to OpenInference, which is OpenTelemetry compatible, and view those traces in [Arize Phoenix](https://github.com/Arize-ai/phoenix).

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
