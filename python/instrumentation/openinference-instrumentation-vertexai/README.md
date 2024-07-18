# OpenInference Google VertexAI Instrumentation

Python auto-instrumentation library for VertexAI SDK and the Google Cloud AI Platform. Traces are fully OpenTelemetry compatible and can be sent to any OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix).

[![pypi](https://badge.fury.io/py/openinference-instrumentation-vertexai.svg)](https://pypi.org/project/openinference-instrumentation-vertexai/)

## Installation

```shell
pip install -Uqqq openinference-instrumentation-vertexai
```

## Quickstart

Install `openinference-instrumentation-vertexai` and `arize-phoenix`.


```shell
pip install -U \
    openinference-instrumentation-vertexai \
    arize-phoenix \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp \
    "opentelemetry-proto>=1.12.0"
```

Start the `phoenix` server so that it is ready to receive traces.
The `phoenix` server runs entirely on your machine and does not send data over the internet.

```shell
python -m phoenix.server.main serve
```

Instrumenting `vertexai` is simple.

```python
from openinference.instrumentation.vertexai import VertexAIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

endpoint = "http://127.0.0.1:4317"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

VertexAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

Now, all calls by `generative_models` are instrumented and can be viewed in the `phoenix` UI.

```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(location="us-central1")
model = GenerativeModel("gemini-1.5-flash")

print(model.generate_content("Why is sky blue?"))
```

## More Info

More details about tracing with OpenInference and `phoenix` can be found in the [`phoenix` documentation](https://docs.arize.com/phoenix).
