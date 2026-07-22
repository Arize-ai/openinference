# OpenInference Cohere Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-cohere.svg)](https://pypi.org/project/openinference-instrumentation-cohere/)

Python auto-instrumentation library for the [Cohere](https://github.com/cohere-ai/cohere-python) Python client.

Chat calls made with the Cohere v2 client (`ClientV2` and `AsyncClientV2`) are traced and exported as OpenInference LLM spans, capturing the input messages, output message, invocation parameters, tool calls, and token counts.

## Installation

```shell
pip install openinference-instrumentation-cohere
```

## Quickstart

Install packages needed for this demonstration.

```shell
pip install openinference-instrumentation-cohere cohere arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start [Phoenix](https://github.com/Arize-ai/phoenix) in the background as a collector. By default, it listens on `http://localhost:6006`.

```shell
phoenix serve
```

Set up `CohereInstrumentor` to trace your application and send the traces to Phoenix.

```python
from openinference.instrumentation.cohere import CohereInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

CohereInstrumentor().instrument(tracer_provider=tracer_provider)
```

Run a chat request. Set the `CO_API_KEY` environment variable with your Cohere API key.

```python
import cohere

co = cohere.ClientV2()
response = co.chat(
    model="command-r-plus",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(response.message.content[0].text)
```

## More Info

- [OpenInference](https://github.com/Arize-ai/openinference)
- [Cohere Python client](https://github.com/cohere-ai/cohere-python)
