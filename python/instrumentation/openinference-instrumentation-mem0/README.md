# OpenInference mem0 Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-mem0.svg)](https://pypi.org/project/openinference-instrumentation-mem0/)

Python auto-instrumentation library for [mem0](https://github.com/mem0ai/mem0) ([`mem0ai`](https://pypi.org/project/mem0ai/)), the memory layer for AI agents.

Memory operations on the open-source `Memory` / `AsyncMemory` classes and on the hosted-platform `MemoryClient` / `AsyncMemoryClient` are traced as OpenInference spans:

| Operation | Span name | Span kind | Recorded |
| --- | --- | --- | --- |
| `search` | `Memory.search` | `RETRIEVER` | `input.value` (the query), `retrieval.documents` (one per memory) |
| `get_all` | `Memory.get_all` | `RETRIEVER` | `input.value` (arguments as JSON), `retrieval.documents` |
| `add` | `Memory.add` | `CHAIN` | `input.value` / `output.value` as JSON |
| `get`, `update`, `delete`, `delete_all`, `history` | `Memory.<operation>` | `CHAIN` | `input.value` / `output.value` as JSON |

Span names use the class that was called, e.g. `AsyncMemory.add` or `MemoryClient.search`.

- **Retrieved memories** become `retrieval.documents.N.document.{id,content,score,metadata}`: `id` is the memory id, `content` is the memory text, `score` is the similarity score (for `search`), and `metadata` holds the remaining fields (`user_id`, `created_at`, `hash`, the memory's own `metadata`, and so on).
- **Scope**: the `user_id`, `agent_id`, `run_id` and `app_id` the call is scoped to (passed directly, inside `filters`, or inside a `MemoryClient` options model) are recorded in the span `metadata`. They are merged with metadata set through `using_metadata`, which takes precedence on conflicting keys.
- **Errors** raised by mem0 (validation errors, LLM or vector-store failures, platform API errors) are recorded on the span with an `ERROR` status and re-raised unchanged.
- LLM, embedding and vector-store calls that mem0 makes internally are not traced by this package. Instrument the provider SDK as well (for example `openinference-instrumentation-openai`) and those spans nest under the mem0 operation that triggered them.

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Supported Features

- `Memory`, `AsyncMemory`, `MemoryClient` and `AsyncMemoryClient`
- Suppressing tracing via `suppress_tracing()`
- Context attribute propagation (`using_session`, `using_user`, `using_attributes`, metadata, tags)
- Sensitive-data masking via `TraceConfig`: `hide_inputs` redacts `input.value`, `hide_outputs` redacts `output.value` and the content and metadata of retrieved memories (`hide_retrieval_documents` does the latter alone)

Requires `mem0ai >= 2.0.0`.

## Installation

```shell
pip install openinference-instrumentation-mem0
```

## Quickstart

```shell
pip install openinference-instrumentation-mem0 mem0ai arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start [Phoenix](https://github.com/Arize-ai/phoenix) as a collector (default `http://localhost:6006`), then:

```python
from mem0 import Memory
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.mem0 import Mem0Instrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

Mem0Instrumentor().instrument(tracer_provider=tracer_provider)

memory = Memory()  # uses OpenAI by default; set OPENAI_API_KEY
memory.add("I love playing tennis on weekends", user_id="alice")
print(memory.search("What does Alice like?", filters={"user_id": "alice"}))
```

See [`examples/`](examples/) for runnable scripts, including one that runs fully offline.

## More Info

- [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
- [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
- [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
