# OpenInference Agno Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-agno.svg)](https://pypi.org/project/openinference-instrumentation-agno/)

Python auto-instrumentation library for Agno Agents

The following instrumentation is fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for monitoring, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix) or [Langfuse](https://langfuse.com).

## Installation

```shell
pip install openinference-instrumentation-agno
```

## Quickstart

This quickstart shows you how to instrument your Agno Agent application.

You've already installed openinference-instrumentation-agno. Next is to install packages for agno,
Phoenix and `opentelemetry-instrument`, which exports traces to it.

```shell
pip install agno arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc opentelemetry-distro
```

Start the Phoenix app in the background as a collector:

```shell
phoenix serve
```

By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address.

The Phoenix app does not send data over the internet. It only operates locally on your machine.

Create a simple Agno agent:

```python example.py
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools

from openinference.instrumentation.agno import AgnoInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

trace_api.set_tracer_provider(tracer_provider=tracer_provider)

# Start instrumenting agno
AgnoInstrumentor().instrument()


agent = Agent(
    model=OpenAIChat(id="gpt-4o-mini"), 
    tools=[DuckDuckGoTools()],
    markdown=True, 
    debug_mode=True,
)

agent.run("What is currently trending on Twitter?")
```

Finally, run the example:

```shell
python example.py
```

Finally, browse for your trace in Phoenix at `http://localhost:6006`!

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)