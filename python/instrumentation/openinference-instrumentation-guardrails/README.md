# OpenInference guardrails Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-guardrails.svg)](https://pypi.org/project/openinference-instrumentation-guardrails/)

Python auto-instrumentation library for LLM applications implemented with Guardrails

Guards are fully OpenTelemetry-compatible and can be sent to an OpenTelemetry collector for monitoring, such as [`arize-phoenix`](https://github.com/Arize-ai/phoenix).

## Installation

```shell
pip install openinference-instrumentation-guardrails
```

## Quickstart

This quickstart shows you how to instrument your guardrailed LLM application 

Install required packages.

```shell
pip install openinference-instrumentation-guardrails guardrails-ai arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start Phoenix in the background as a collector. By default, it listens on `http://localhost:6006`. You can visit the app via a browser at the same address. (Phoenix does not send data over the internet. It only operates locally on your machine.)

```shell
python -m phoenix.server.main serve
```

Install the TwoWords validator that's used in the Guard.

```shell
guardrails hub install hub://guardrails/two_words
```

Set up `GuardrailsInstrumentor` to trace your guardrails application and sends the traces to Phoenix at the endpoint defined below.
```python
from openinference.instrumentation.guardrails import GuardrailsInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
import os

os.environ["OPENAI_API_KEY"] = "YOUR_KEY_HERE"

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
trace_api.set_tracer_provider(tracer_provider)

GuardrailsInstrumentor().instrument()
```

Set up a simple example of LLM call using a Guard
```python
from guardrails import Guard
from guardrails.hub import TwoWords
import openai

guard = Guard().use(
    TwoWords(),
)

response = guard(
    llm_api=openai.chat.completions.create,
    prompt="What is another name for America?",
    model="gpt-3.5-turbo",
    max_tokens=1024,
)

print(response)
```

## More Info

* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)