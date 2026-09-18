# OpenInference TypeSafe AI Instrumentation

[![pypi](https://badge.fury.io/py/openinference-instrumentation-typesafe.svg)](https://pypi.org/project/openinference-instrumentation-typesafe/)

Python auto-instrumentation library for the [TypeSafe AI](https://docs.typesafe.ai/) Python SDK ([`typesafe-sdk`](https://pypi.org/project/typesafe-sdk/)).

Calls to `TypeSafeClient.system_one` and `AsyncTypeSafeClient.system_one` are traced and exported as OpenInference LLM spans. A System One request sends a `state` plus a map of typed `questions` (Noul, Choice, Score) and returns one typed `answer` per question, so the span records:

- `input.value`: the request body (`state`, `model`, `questions`) as JSON
- `llm.invocation_parameters`: the `model` and the `questions` map, which acts as the response schema
- `output.value`: the response body (`model`, `answers`, `usage`) as JSON
- `llm.request.model_name` (for example `jev-latest`) and `llm.response.model_name` (the resolved model, for example `jev-1.13.0`)
- `llm.token_count.prompt`, `llm.token_count.completion`, and `llm.token_count.total`

A System One call is not a chat exchange, so the `state` and the `answers` are recorded only as `input.value` and `output.value`, not as `llm.input_messages` / `llm.output_messages`.

Because the `questions` map rides in `llm.invocation_parameters`, it is masked by `hide_llm_invocation_parameters`, not by `hide_inputs`. The `state` — the caller data the questions are asked about — is only ever recorded in `input.value`, so `TraceConfig(hide_inputs=True)` is enough to keep it off the span. Use `TraceConfig(hide_inputs=True, hide_outputs=True, hide_llm_invocation_parameters=True)` when the question instructions themselves are sensitive too.

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

## Supported Features

- Synchronous and asynchronous clients (`TypeSafeClient` and `AsyncTypeSafeClient`)
- All three question primitives, passed as SDK objects or raw dictionaries
- Suppressing tracing via `suppress_tracing()`
- Context attribute propagation (`using_session`, `using_user`, `using_attributes`, metadata, tags)
- Sensitive-data masking via `TraceConfig` (e.g. `hide_inputs`, `hide_outputs`, `hide_llm_invocation_parameters`)

Requires `typesafe-sdk >= 0.6.0`.

## Installation

```shell
pip install openinference-instrumentation-typesafe
```

## Quickstart

```shell
pip install openinference-instrumentation-typesafe typesafe-sdk arize-phoenix opentelemetry-sdk opentelemetry-exporter-otlp
```

Start [Phoenix](https://github.com/Arize-ai/phoenix) as a collector (default `http://localhost:6006`), then:

```python
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from openinference.instrumentation.typesafe import TypeSafeAIInstrumentor

endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

TypeSafeAIInstrumentor().instrument(tracer_provider=tracer_provider)
```

Make a System One call. Set the `TYPESAFE_API_KEY` environment variable with your key.

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

client = TypeSafeClient()
response = client.system_one(
    state={"document": "I was charged twice. Please fix this ASAP."},
    questions={
        "billing": Noul(instructions="Is this ticket about billing?"),
        "tone": Choice(
            instructions="What is the customer's tone?",
            criteria={"calm": None, "frustrated": None, "angry": None},
        ),
        "urgency": Score(
            instructions="How urgent is this ticket?",
            criteria=["can wait", "this week", "today"],
        ),
    },
)
print(response.nouls["billing"].noul)
print(response.choices["tone"].choice)
print(response.scores["urgency"].score)
```

Runnable examples, including async usage and context attributes, are in the [`examples/`](./examples) directory.

## More Info

- [More info on OpenInference and Phoenix](https://docs.arize.com/phoenix)
- [How to customize spans to track sessions, metadata, etc.](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#customizing-spans)
- [How to account for private information and span payload customization](https://github.com/Arize-ai/openinference/tree/main/python/openinference-instrumentation#tracing-configuration)
