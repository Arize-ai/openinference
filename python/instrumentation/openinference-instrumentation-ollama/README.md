# OpenInference Ollama Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-ollama.svg)](https://pypi.org/project/openinference-instrumentation-ollama/)

Python auto-instrumentation library for the [Ollama](https://github.com/ollama/ollama) Python client.

Chat calls made with the `ollama` package are traced and exported as OpenInference LLM spans, capturing the input messages, output message, invocation parameters, tool calls, and token counts. Works with `ollama.chat`, `ollama.Client`, and `ollama.AsyncClient`.

## Installation

```shell
pip install openinference-instrumentation-ollama
```

## Quickstart

Install packages needed for this demonstration.

```shell
pip install openinference-instrumentation-ollama ollama arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start [Phoenix](https://github.com/Arize-ai/phoenix) in the background as a collector. By default, it listens on `http://localhost:6006`.

```shell
phoenix serve
```

Set up `OllamaInstrumentor` to trace your application and send the traces to Phoenix.

```python
from openinference.instrumentation.ollama import OllamaInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

OllamaInstrumentor().instrument(tracer_provider=tracer_provider)
```

Run a chat completion against a locally running Ollama server.

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(response.message.content)
```

## More Info

- [OpenInference](https://github.com/Arize-ai/openinference)
- [Ollama Python client](https://github.com/ollama/ollama-python)
