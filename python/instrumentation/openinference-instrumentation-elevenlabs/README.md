# OpenInference ElevenLabs Instrumentation

Python auto-instrumentation library for ElevenLabs. This library allows you to convert ElevenLabs traces to OpenInference, which is OpenTelemetry compatible, and view those traces in [Arize Phoenix](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-elevenlabs
```

## Quickstart

This quickstart shows you how to setup tracing in your ElevenLabs application:

```python
from phoenix.otel import register
from openinference.instrumentation.elevenlabs import ElevenLabsInstrumentor

# Set up the tracer provider
tracer_provider = register(
    project_name="default"  # Phoenix project name
)

# Add auto-instrumentor at the top of the application
ElevenLabsInstrumentor().instrument(tracer_provider=tracer_provider)

# Use ElevenLabs as usual
# ... (add your ElevenLabs code here)
```

After configuring tracing, ElevenLabs API calls in the running application are logged to your project in Phoenix or Arize AX.
