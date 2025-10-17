# OpenInference Pipecat Instrumentation

Python auto-instrumentation library for Pipecat. This library allows you to convert Pipecat traces to OpenInference, which is OpenTelemetry compatible, and view those traces in [Arize Phoenix](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-pipecat
```

## Quickstart

This quickstart shows you how to view your Pipecat traces in Phoenix.

Install required packages.

```shell
pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp pipecat-ai
```

Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)

```shell
phoenix serve
```

Here's a simple example that demonstrates how to convert Pipecat traces into OpenInference and view those traces in Phoenix:

```python
import os
import grpc
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from phoenix.otel import register
from openinference.instrumentation.pipecat import OpenInferenceSpanProcessor
from pipecat.utils.tracing import setup_tracing

# Set your API keys
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Set up the tracer provider
tracer_provider = register(
    project_name="default" #Phoenix project name
)

tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
    
tracer_provider.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="http://localhost:4317", #if using phoenix cloud, change to phoenix cloud endpoint (phoenix cloud space -> settings -> endpoint/hostname)
            headers={},
            compression=grpc.Compression.Gzip,  # use enum instead of string
        )
    )
)

# Initialize Pipecat tracing
setup_tracing(
    service_name="pipecat-phoenix-demo",
    exporter=OTLPSpanExporter(
        endpoint="http://localhost:4317",
        headers={},
        compression=grpc.Compression.Gzip,
    ),
)

# Build your Pipecat pipeline
# ... (add your Pipecat pipeline code here)

# Now view your converted Pipecat traces in Phoenix!
```
## This example:

1. Uses Pipecat's built-in tracing utilities to instrument the application.
2. Defines a Pipecat pipeline for voice/conversational AI
3. Traces are exported to Phoenix using the span processor. 

The traces will be visible in the Phoenix UI at `http://localhost:6006`.

## More Info

-   [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
-   [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
-   [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
