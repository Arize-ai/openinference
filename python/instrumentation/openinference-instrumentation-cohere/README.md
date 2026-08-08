# OpenInference Cohere Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-cohere.svg)](https://pypi.org/project/openinference-instrumentation-cohere/)

Python auto-instrumentation library for the [Cohere](https://github.com/cohere-ai/cohere-python) Python client.

Chat, embedding, and rerank calls made with the Cohere v2 client (`ClientV2` and
`AsyncClientV2`) are traced and exported as OpenInference spans. Chat spans capture messages,
invocation parameters, tool calls, and token counts. Embedding spans capture the model name,
input text, invocation parameters, vectors, and token counts. Rerank spans capture the query,
model, input documents, ranked output documents, and relevance scores.

## Coverage

`ClientV2.chat`, `AsyncClientV2.chat`, `ClientV2.chat_stream`,
`AsyncClientV2.chat_stream`, `ClientV2.embed`, `AsyncClientV2.embed`, `ClientV2.rerank`, and
`AsyncClientV2.rerank` are instrumented.
Streamed calls finish their span when the returned iterator is exhausted, with the accumulated
output message, tool calls, and token counts.

The following are **not** traced, and calls to them produce no spans:

- The v1 client (`cohere.Client`)
- Classify endpoints

## Installation

```shell
pip install openinference-instrumentation-cohere
```

PyPI package: [`openinference-instrumentation-cohere`](https://pypi.org/project/openinference-instrumentation-cohere/)

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
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(response.message.content[0].text)
```

## More Info

- [OpenInference](https://github.com/Arize-ai/openinference)
- [Cohere Python client](https://github.com/cohere-ai/cohere-python)
