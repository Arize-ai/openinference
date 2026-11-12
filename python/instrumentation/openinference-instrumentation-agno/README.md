# OpenInference Agno Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-agno.svg)](https://pypi.org/project/openinference-instrumentation-agno/)

Python auto-instrumentation library for Agno Agents

The following instrumentation is fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for monitoring, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix), [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference), or [Langfuse](https://langfuse.com).

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

agent.print_response("What is currently trending on Twitter?")
```

Finally, run the example:

```shell
python example.py
```

Finally, browse for your trace in Phoenix at `http://localhost:6006`!

## Isolating agno spans from the host context

By default, agno spans become the current span in OpenTelemetry's global context, so they nest under whatever span the host application has active. Pass `runtime_context` to keep them in a separate context instead:

```python
from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext

AgnoInstrumentor().instrument(
    tracer_provider=tracer_provider,
    runtime_context=ContextVarsRuntimeContext(),
)
```

Any object with `attach`, `get_current` and `detach`, such as an OpenTelemetry `_RuntimeContext`, works. Agno spans then form their own trace and never replace the host's current span. Code that agno calls, such as a tool, still sees the host's span as current.

Only agno's own spans use the injected context. Spans from other instrumentors that run inside agno, such as `OpenAIInstrumentor`, still read the global context, so they parent to the host span rather than to the agno span that made the call.

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [More info on OpenInference and Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)