# OpenInference Google GenAI Instrumentation

Python auto-instrumentation library for GenAI SDK. Traces are fully OpenTelemetry compatible and can be sent to any OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Installation

```shell
pip install -Uqqq openinference-instrumentation-google-genai
```

## Quickstart

Install `openinference-instrumentation-google-genai` and `arize-phoenix`.

```shell
pip install -U \
    openinference-instrumentation-google-genai \
    arize-phoenix \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    "opentelemetry-proto>=1.12.0"
```

Start the `phoenix` server so that it is ready to receive traces.
The `phoenix` server runs entirely on your machine and does not send data over the internet.

```shell
phoenix serve
```

Instrumenting `genai` is simple.

```python
from openinference.instrumentation.google_genai import GoogleGenAIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:4317"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

GoogleGenAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

Now, all calls by `generate_content` are instrumented and can be viewed in the `phoenix` UI.

## Supported APIs

This instrumentor traces the Google GenAI SDK, including `generate_content` (and its
streaming variant), `embed_content`, `Caches.create`, the Interactions resource, Live
sessions, and the **Gemini Decisions API** (`client.decisions.create`), which is traced as
an OpenInference `DECISION` span. The Decisions API requires `google-genai >= 3.2` and the
`v1beta` API version (`genai.Client(http_options={"api_version": "v1beta"})`); on older SDK
versions, which have no decisions resource, that tracing is skipped and the rest of the SDK
is still instrumented. See [`examples/decisions.py`](examples/decisions.py) for a runnable
example.

## Progress

This instrumentation is a work in progress

-   [x] parse messages and invocation
-   [ ] capture tool definitions

## More Info

-   [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
-   [More info on OpenInference and Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference)
-   [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
-   [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
