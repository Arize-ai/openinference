# OpenInference Together AI Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-together.svg)](https://pypi.org/project/openinference-instrumentation-together/)

Python auto-instrumentation library for the [Together AI](https://github.com/togethercomputer/together-python) Python client.

Chat completion calls made with the `together` client (`Together` and `AsyncTogether`) are traced and exported as OpenInference LLM spans, capturing the input messages, output messages, invocation parameters, tool calls, streaming output, and token counts.

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/docs/ax).

## Supported Features

- Synchronous and asynchronous chat completions (`Together` and `AsyncTogether`)
- Streaming (`stream=True`): the span stays open until the stream is consumed, and the accumulated output, tool calls, and token counts are recorded from the streamed chunks
- Tool calls, captured on both requests and responses
- Suppressing tracing via `suppress_tracing()`
- Context attribute propagation (`using_session`, `using_user`, `using_attributes`, metadata, tags)
- Sensitive-data masking via `TraceConfig` (e.g. `hide_inputs`)

Requires `together >= 2.0.0`.

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

Streaming works the same way — the span is finished when the stream is exhausted:

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Write a haiku about observability."}],
    stream=True,
)
for chunk in stream:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

Runnable examples — including async usage, streaming with a reasoning model, and tool calls — are in the [`examples/`](./examples) directory.

## More Info

- [OpenInference](https://github.com/Arize-ai/openinference)
- [Together AI Python client](https://github.com/togethercomputer/together-python)
- [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
- [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
- [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
- [More info on OpenInference and Arize AX](https://arize.com/docs/ax)
