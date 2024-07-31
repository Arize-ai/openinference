# OpenInference liteLLM Instrumentation

Python autoinstrumentation library for the liteLLM package

This package implements OpenInference tracing for the following liteLLM functions:
- completion()
- acompletion()
- completion_with_retries()
- embedding()
- aembedding()
- image_generation()
- aimage_generation()

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize `phoenix`](https://github.com/Arize-ai/phoenix).


## Installation

```shell
pip install openinference-instrumentation-litellm
```

## Quickstart

In a notebook environment (`jupyter`, `colab`, etc.) install `openinference-instrumentation-litellm` and `arize-phoenix`.


```shell
pip install openinference-instrumentation-litellm arize-phoenix litellm
```

First, import dependencies required to autoinstrument `liteLLM` and set up `phoenix` as an collector for OpenInference traces.

```python
import litellm
import phoenix as px

from openinference.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
```

Next, we'll start a `phoenix` server and set it as a collector.

```python
px.launch_app()
session_url = px.active_session().url
phoenix_otlp_endpoint = urljoin(session_url, "v1/traces")
phoenix_exporter = OTLPSpanExporter(endpoint=phoenix_otlp_endpoint)
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=phoenix_exporter))
trace_api.set_tracer_provider(tracer_provider=tracer_provider)
```

Instrumenting `litellm` is simple:

```python
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
```

Now, all calls to `litellm` functions are instrumented and can be viewed in the `phoenix` UI.

```python
completion_response = litellm.completion(model="gpt-3.5-turbo", 
                   messages=[{"content": "What's the capital of China?", "role": "user"}])
print(completion_response)
```

```python
completion_response = litellm.completion(model="gpt-3.5-turbo", 
                   messages=[{"content": "What's the capital of China?", "role": "user"}])
print(completion_response)
```

```python
acompletion_response = await litellm.acompletion(
            model="gpt-3.5-turbo",
            messages=[{ "content": "Hello, I want to bake a cake","role": "user"},
                      { "content": "Hello, I can pull up some recipes for cakes.","role": "assistant"},
                      { "content": "No actually I want to make a pie","role": "user"},],
            temperature=0.7,
            max_tokens=20
        )
print(acompletion_response)
```

```python
embedding_response = litellm.embedding(model='text-embedding-ada-002', input=["good morning!"])
print(embedding_response)
```

```python
image_gen_response = litellm.image_generation(model='dall-e-2', prompt="cute baby otter")
print(image_gen_response)
```

## More Info

More details about tracing with OpenInference and `phoenix` can be found in the [`phoenix` documentation](https://docs.arize.com/phoenix).
