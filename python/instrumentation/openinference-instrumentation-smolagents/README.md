# OpenInference smolagents Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-smolagents.svg)](https://pypi.org/project/openinference-instrumentation-smolagents/)

Python auto-instrumentation library for LLM agents implemented with smolagents

Crews are fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for monitoring, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-smolagents
```

## Quickstart

This quickstart shows you how to instrument your LLM agent application.

You've already installed openinference-instrumentation-smolagents. Next is to install packages for smolagents,
Phoenix and `opentelemetry-instrument`, which exports traces to it.

```shell
pip install smolagents arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp-proto-grpc opentelemetry-distro
```

Start Phoenix in the background as a collector, which listens on `http://localhost:6006` and default gRPC port 4317.
Note that Phoenix does not send data over the internet. It only operates locally on your machine.

```shell
python -m phoenix.server.main serve
```

Create an example like this:

```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("How many seconds would it take for a leopard at full speed to run through Pont des Arts?")
```

Then, run it like this:

```shell
opentelemetry-instrument python example.py
```

Finally, browse for your trace in Phoenix at `http://localhost:6006`!

## Manual instrumentation

`opentelemetry-instrument` is the [Zero-code instrumentation](https://opentelemetry.io/docs/zero-code/python) approach
for Python. It avoids explicitly importing and configuring OpenTelemetry code in your main source. Alternatively, you
can copy-paste the following into your main source and run it without `opentelemetry-instrument`.

```python
from opentelemetry.sdk.trace import TracerProvider

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(otlp_exporter))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
```

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)