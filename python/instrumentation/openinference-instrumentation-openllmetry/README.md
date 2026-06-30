# OpenInference OpenLLMetry (Traceloop)


Python auto-instrumentation library for OpenLLMetry. This library allows you to convert OpenLLMetry traces to OpenInference, which is OpenTelemetry compatible, and view those traces in [Arize Phoenix](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-openllmetry
```

## Quickstart

This quickstart shows you how to view your OpenLLMetry traces in Phoenix.

Install required packages.

```shell
pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-openai
```

Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)

```shell
phoenix serve
```

Here's a simple example that demonstrates how to view convert OpenLLMetry traces into OpenInference and view those traces in Phoenix:

```python
import os
import grpc
import openai
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from phoenix.otel import register
from openinference.instrumentation.openllmetry import OpenInferenceSpanProcessor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor

# Set your OpenAI API key
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


OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

# Define and invoke your OpenAI model
client = openai.OpenAI()

messages = [
        {"role": "user", "content": "What is the national food of Yemen?"}
    ]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
)

# Now view your converted OpenLLMetry traces in Phoenix!
```
## This example:

1. Uses OpenLLMetry Instrumentor to instrument the application.
2. Defines a simple OpenAI model and runs a query
3. Queries are exported to Phoenix using a span processor. 

The traces will be visible in the Phoenix UI at `http://localhost:6006`.

## More Info

-   [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
-   [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
-   [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
