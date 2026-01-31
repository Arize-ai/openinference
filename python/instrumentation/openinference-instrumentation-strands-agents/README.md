# OpenInference Strands Agents Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-strands-agents.svg)](https://pypi.org/project/openinference-instrumentation-strands-agents/)

Python instrumentation library for Strands Agents.

This package provides a span processor that transforms Strands' native OpenTelemetry spans to OpenInference format. The traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-strands-agents
```

## Quickstart

In this example we will instrument a small Strands agent program and observe the traces via [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

Install packages.

```shell
pip install openinference-instrumentation-strands-agents strands-agents arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the phoenix server so that it is ready to collect traces.
The Phoenix server runs entirely on your machine and does not send data over the internet.

```shell
python -m phoenix.server.main serve
```

In a python file, setup the Strands telemetry and add the `StrandsAgentsToOpenInferenceProcessor` to transform spans.

```python
import os

from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.telemetry import StrandsTelemetry

from openinference.instrumentation.strands_agents import StrandsAgentsToOpenInferenceProcessor

# Setup Strands native telemetry
telemetry = StrandsTelemetry()
telemetry.setup_otlp_exporter(endpoint="http://127.0.0.1:6006/v1/traces")

# Add OpenInference processor to transform spans
telemetry.tracer_provider.add_span_processor(StrandsAgentsToOpenInferenceProcessor())


@tool
def get_weather(city: str) -> dict:
    """Get the current weather for a city.

    Args:
        city: The name of the city
    """
    return {
        "status": "success",
        "content": [{"text": f"The weather in {city} is sunny and 72°F."}],
    }


if __name__ == "__main__":
    # Create and run agent
    model = OpenAIModel(model_id="gpt-4o-mini")
    agent = Agent(
        name="WeatherAgent",
        model=model,
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant.",
    )

    result = agent("What's the weather in San Francisco?")
    print(result)
```

Since we are using OpenAI, we must set the `OPENAI_API_KEY` environment variable to authenticate with the OpenAI API.

```shell
export OPENAI_API_KEY=your-api-key
```

Now simply run the python file and observe the traces in Phoenix.

```shell
python your_file.py
```

## Important: Processor Ordering

The `StrandsAgentsToOpenInferenceProcessor` **mutates spans in-place**. This means the order in which you add span processors matters.

Add the `StrandsAgentsToOpenInferenceProcessor` **before** any exporters that should receive the transformed OpenInference spans:

```python
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Correct order: processor first, then exporter
telemetry.tracer_provider.add_span_processor(StrandsAgentsToOpenInferenceProcessor())
telemetry.tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
```

If you need to export both the original GenAI spans and transformed OpenInference spans to different destinations, you'll need to set up the processors carefully or consider using separate tracer providers.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
