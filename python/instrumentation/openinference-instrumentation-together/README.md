# OpenInference Together AI Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-together.svg)](https://pypi.org/project/openinference-instrumentation-together/)

Python auto-instrumentation library for the [Together AI](https://github.com/togethercomputer/together-python) Python client.

Chat completion calls made with the `together` client (`Together` and `AsyncTogether`) are traced and exported as OpenInference LLM spans, capturing the input messages, output messages, invocation parameters, tool calls, and token counts.

## Installation

```shell
pip install openinference-instrumentation-together
```

## Quickstart

```shell
pip install openinference-instrumentation-together together arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start [Phoenix](https://github.com/Arize-ai/phoenix) as a collector (default `http://localhost:6006`), then:

```python
from openinference.instrumentation.together import TogetherInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

TogetherInstrumentor().instrument(tracer_provider=tracer_provider)
```

Run a chat completion. Set the `TOGETHER_API_KEY` environment variable with your key.

```python
from together import Together

client = Together()
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(response.choices[0].message.content)
```

## More Info

- [OpenInference](https://github.com/Arize-ai/openinference)
- [Together AI Python client](https://github.com/togethercomputer/together-python)
