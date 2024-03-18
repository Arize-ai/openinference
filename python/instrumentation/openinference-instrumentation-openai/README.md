# OpenInference OpenAI Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-openai.svg)](https://pypi.org/project/openinference-instrumentation-openai/)

Python auto-instrumentation library for OpenAI's python SDK.

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix)

## Installation

```shell
pip install openinference-instrumentation-openai
```

## Quickstart

In this example we will instrument a small program that uses OpenAI and observe the traces via [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

Install packages.

```shell
pip install openinference-instrumentation-openai openai arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start the phoenix server so that it is ready to collect traces.
The Phoenix server runs entirely on your machine and does not send data over the internet.

```shell
python -m phoenix.server.main serve
```

In a python file, setup the `OpenAIInstrumentor` and configure the tracer to send traces to Phoenix.

```python
import openai
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# Optionally, you can also print the spans to the console.
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace_api.set_tracer_provider(tracer_provider)

OpenAIInstrumentor().instrument()


if __name__ == "__main__":
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Write a haiku."}],
        max_tokens=20,
    )
    print(response.choices[0].message.content)

```

Since we are using OpenAI, we must set the `OPENAI_API_KEY` environment variable to authenticate with the OpenAI API.

```shell
export OPENAI_API_KEY=your-api-key
```

Now simply run the python file and observe the traces in Phoenix.

```shell
python your_file.py
```

## More Info

Fore details about tracing with OpenInference and Phoenix, consult the [Phoenix documentation](https://docs.arize.com/phoenix).

For AI/ML observability solutions in production, check out the docs on [Arize](https://docs.arize.com/arize).
