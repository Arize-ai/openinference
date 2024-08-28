# OpenInference LiteLLM Instrumentation

[LiteLLM](https://github.com/BerriAI/litellm) allows developers to call all LLM APIs using the openAI format. [LiteLLM Proxy](https://docs.litellm.ai/docs/simple_proxy) is a proxy server to call 100+ LLMs in OpenAI format. Both are supported by this auto-instrumentation.

This package implements OpenInference tracing for the following LiteLLM functions:
- completion()
- acompletion()
- completion_with_retries()
- embedding()
- aembedding()
- image_generation()
- aimage_generation()

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix).


## Installation

```shell
pip install openinference-instrumentation-litellm
```

## Quickstart

In a notebook environment (`jupyter`, `colab`, etc.) install `openinference-instrumentation-litellm` if you haven't already as well as `arize-phoenix` and `litellm`.


```shell
pip install openinference-instrumentation-litellm arize-phoenix litellm
```

First, import dependencies required to autoinstrument `liteLLM` and set up `phoenix` as an collector for OpenInference traces.

```python
import litellm
import phoenix as px

from openinference.instrumentation.litellm import LiteLLMInstrumentor

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
```

Next, we'll start a `phoenix` server and set it as a collector.

```python
session = px.launch_app()
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
```

Set up any API keys needed in you API calls. For example:

```python
import os
os.environ["OPENAI_API_KEY"] = "PASTE_YOUR_API_KEY_HERE"
```

Instrumenting `LiteLLM` is simple:

```python
LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
```

Now, all calls to `LiteLLM` functions are instrumented and can be viewed in the `phoenix` UI.

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

You can also uninstrument the functions as follows
```python
LiteLLMInstrumentor().uninstrument(tracer_provider=tracer_provider)
```
Now any liteLLM function calls you make will not send traces to Phoenix until instrumented again

## More Info

* Details on how to setup a [LiteLLM Proxy](https://docs.litellm.ai/docs/observability/arize_integration)
* [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
* [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
* [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
