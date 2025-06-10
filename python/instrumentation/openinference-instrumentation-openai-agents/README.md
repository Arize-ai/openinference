# OpenInference OpenAI Agents Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-openai-agents.svg)](https://pypi.org/project/openinference-instrumentation-openai-agents/)

Python auto-instrumentation library for OpenAI Agents python SDK.

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix)

## Installation

```shell
pip install openinference-instrumentation-openai-agents
```

## Quickstart

In this example we will instrument a small program that uses OpenAI and observe the traces via [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

Install packages.

```shell
pip install openinference-instrumentation-openai-agents arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the phoenix server so that it is ready to collect traces.
The Phoenix server runs entirely on your machine and does not send data over the internet.

```shell
phoenix serve
```

In a python file, set up the `OpenAIAgentsInstrumentor` and configure the tracer to send traces to Phoenix.

```python
from agents import Agent, Runner
from openinference.instrumentation.openai_agents import OpenAIAgentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

OpenAIAgentsInstrumentor().instrument(tracer_provider=tracer_provider)


agent = Agent(name="Assistant", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

Since we are using OpenAI, we must set the `OPENAI_API_KEY` environment variable to authenticate with the OpenAI API.

```shell
export OPENAI_API_KEY=your-api-key
```

Now simply run the python file and observe the traces in Phoenix.

```shell
python your_file.py
```
