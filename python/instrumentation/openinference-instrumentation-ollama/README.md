# OpenInference Ollama Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-ollama.svg)](https://pypi.org/project/openinference-instrumentation-ollama/)

Python auto-instrumentation library for the [Ollama Python client](https://github.com/ollama/ollama-python).

The traces emitted by this instrumentation are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/docs/ax).

## What is instrumented

`chat` calls made through `ollama.chat`, `ollama.Client.chat`, and `ollama.AsyncClient.chat` are exported as OpenInference LLM spans (named `chat` and `async_chat` respectively), capturing:

- Input and output messages (`llm.input_messages.*`, `llm.output_messages.*`), including tool calls
- Streaming (`stream=True`): the span finishes when the stream is exhausted, fails, or is abandoned, with the output message and token counts reconstructed from the accumulated chunks
- Tool definitions as `llm.tools.N.tool.json_schema` — plain Python functions passed via `tools=[...]` are converted to their JSON schemas
- `llm.provider` (`ollama`) and `llm.model_name` (recorded from the request as well, so errored calls still carry the model)
- Token counts: `prompt_eval_count` → `llm.token_count.prompt`, `eval_count` → `llm.token_count.completion`, with the total derived when both are present
- `llm.invocation_parameters` (request options other than `messages`, `model`, and `tools`)
- Errors: exceptions set the span status to `ERROR` and are recorded as span events

Not currently instrumented: `generate`, `embed`/`embeddings`, and other client methods.

> [!NOTE]
> Call `OllamaInstrumentor().instrument()` before making chat calls, and invoke chat via `import ollama; ollama.chat(...)` or a `Client`/`AsyncClient` instance. A reference captured before instrumentation (e.g. `from ollama import chat` at import time) keeps the uninstrumented function and produces no spans. To be captured on the span, `tools` must be a list or tuple (not a generator).

Context attributes (session, user, metadata, tags via [`using_attributes`](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)) propagate onto spans, and sensitive data can be masked with a [`TraceConfig`](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration), e.g. `OllamaInstrumentor().instrument(tracer_provider=tracer_provider, config=TraceConfig(hide_inputs=True))`. Calls made inside `with suppress_tracing():` are not traced.

## Installation

```shell
pip install openinference-instrumentation-ollama
```

Requires `ollama >= 0.4.0`.

## Quickstart

Install packages needed for this demonstration.

```shell
pip install openinference-instrumentation-ollama ollama arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Install and start the [Ollama server](https://ollama.com/download) (`pip install ollama` installs only the client), then pull a model. The server listens on `http://localhost:11434` by default.

```shell
ollama pull llama3.2
```

Start [Phoenix](https://github.com/Arize-ai/phoenix) in the background as a collector. By default, it listens on `http://localhost:6006`. (Phoenix does not send data over the internet. It only operates locally on your machine.)

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

Run a chat completion against the locally running Ollama server.

```python
import ollama

response = ollama.chat(
    model="llama3.2",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
)
print(response.message.content)
```

Now view your traces in the Phoenix UI at `http://localhost:6006`.

## Examples

The [`examples/`](./examples/) directory contains runnable scripts. They require a running Phoenix and Ollama server, read the model from the `OLLAMA_MODEL` environment variable (default `llama3.2`), and send traces to a Phoenix project named `ollama-examples`.

```shell
pip install -r examples/requirements.txt
OLLAMA_MODEL=llama3.2 python examples/chat.py
```

| Example                                                | Description                                                    |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| [`chat.py`](./examples/chat.py)                        | A basic chat completion                                        |
| [`streaming_and_tools.py`](./examples/streaming_and_tools.py) | Streaming with session attributes, and tool calling with a plain Python function |

## Development

From the `python/` directory: `tox run -e test-ollama` runs the tests, and `tox run -e ruff-mypy-test-ollama` runs all checks.

## More Info

- [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
- [More info on OpenInference and Arize AX](https://arize.com/docs/ax)
- [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
- [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
- [Ollama Python client](https://github.com/ollama/ollama-python)
