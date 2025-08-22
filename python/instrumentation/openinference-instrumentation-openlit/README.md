# OpenInference OpenLit Instrumentation

Python auto-instrumentation library for OpenLIT. This library allows you to convert OpenLIT traces to OpenInference, which is OpenTelemetry compatible, and view those traces in [Arize Phoenix](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-openlit
```

## Quickstart

This quickstart shows you how to view your OpenLIT traces in Phoenix.

Install required packages.

```shell
pip install arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp openlit semantic-kernel
```

Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address.

```shell
phoenix serve
```

Here's a simple example that demonstrates how to convert OpenLIT traces into OpenInference and view those traces in Phoenix:

```python
import os
import grpc
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from phoenix.otel import register
from openinference.instrumentation.openlit import OpenInferenceSpanProcessor
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import openlit

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

# Initialize OpenLit tracer
tracer = tracer_provider.get_tracer(__name__)
openlit.init(tracer=tracer)

# Set up Semantic Kernel with OpenLIT
kernel = Kernel()
kernel.add_service(
    OpenAIChatCompletion(
        service_id="default",
        ai_model_id="gpt-4o-mini",
    ),
)

# Define and invoke your model
result = await kernel.invoke_prompt(
    prompt="What is the national food of Yemen?",
    arguments={},
)

# Now view your converted OpenLIT traces in Phoenix!
```

## This example:

1. Uses OpenLIT Instrumentor to instrument the application.
2. Defines a simple Semantic Kernel model and runs a query
3. Queries are exported to Phoenix using a span processor. 

The traces will be visible in the Phoenix UI at `http://localhost:6006`.

## More Info

-   [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
-   [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
-   [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration) 