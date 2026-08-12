# OpenInference PydanticAI

[![pypi](https://badge.fury.io/py/openinference-instrumentation-pydantic-ai.svg)](https://pypi.org/project/openinference-instrumentation-pydantic-ai/)

Python auto-instrumentation library for PydanticAI. These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Installation

```shell
pip install "openinference-instrumentation-pydantic-ai[instruments]"
```

### Pydantic AI compatibility

The default `instruments` extra supports Pydantic AI 2.24.0 and newer with
genai-prices 0.1 or newer. Pydantic AI 2.17.0 first added support for the additional
usage fields in genai-prices 0.1; the supported floor is 2.24.0 so new installations
also include Pydantic AI's subsequent security fixes.

Supported Pydantic AI V1 releases must use genai-prices below 0.1. With genai-prices
0.1 or newer, Pydantic AI V1 can silently omit token counts from its spans.
Pydantic AI 2.0.0 through 2.23.x are outside the supported ranges; upgrade to 2.24.0
or newer.

| Pydantic AI version | Compatible genai-prices version |
|---|---|
| `>=1.107.2,<2.0.0` | `>=0.0.62,<0.1` |
| `>=2.24.0` | `>=0.1` |

Install the `instruments-legacy` extra to use a supported Pydantic AI V1 release with
a safe genai-prices constraint:

```shell
pip install "openinference-instrumentation-pydantic-ai[instruments-legacy]"
```

## Quickstart

This quickstart shows you how to instrument your PydanticAI agents.

Install required packages.

```shell
pip install "pydantic-ai>=2.24.0" "genai-prices>=0.1.0" arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)

```shell
phoenix serve
```

Here's a simple example that demonstrates how to use PydanticAI with OpenInference instrumentation:

```python
import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "YOUR_OPENAI_API_KEY"

# Set up the tracer provider
tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)

# Add the OpenInference span processor
endpoint = "http://127.0.0.1:6006/v1/traces"
exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))


# Define your Pydantic model
class LocationModel(BaseModel):
    city: str
    country: str

instrumentation = InstrumentationSettings(version=2)

# Create and configure the agent
model = OpenAIModel("gpt-4", provider=OpenAIProvider())
agent = Agent(model, output_type=LocationModel, instrument=instrumentation)

# Run the agent
result = agent.run_sync("The windy city in the US of A.")
print(result)
```

This example:

1. Sets up OpenTelemetry tracing with Phoenix
2. Defines a simple Pydantic model for location data
3. Creates a PydanticAI agent with instrumentation enabled
4. Runs a query and gets structured output

The traces will be visible in the Phoenix UI at `http://localhost:6006`.

## More Info

-   [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
-   [More info on OpenInference and Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)
-   [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
-   [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
