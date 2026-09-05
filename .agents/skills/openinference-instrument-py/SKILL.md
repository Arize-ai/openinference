---
name: openinference-instrument-py
description: >
  Instrument Python LLM, agent, and RAG code with OpenInference on OpenTelemetry: set up a
  TracerProvider with a project name, register auto-instrumentors (OpenAI, Anthropic,
  LangChain, LlamaIndex, and others), trace your own functions with OITracer decorators
  (chain, agent, tool, llm) or manual spans, build attributes with helpers, attach context
  attributes (using_session, using_user, using_metadata, using_tags), mask data with
  TraceConfig, pause tracing with suppress_tracing, and follow the checklist for writing a
  new instrumentor in this repo. Use for any Python tracing question involving OpenInference,
  Phoenix, or Arize.
---

# OpenInference for Python

Concepts live in the `openinference` skill; read it first. Packages:
`openinference-instrumentation` (core helpers), `openinference-semantic-conventions`
(attribute constants), `openinference-instrumentation-<lib>` (one auto-instrumentor per library).

## Setup

```python
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from openinference.instrumentation import TracerProvider, TraceConfig
from openinference.semconv.resource import ResourceAttributes

tracer_provider = TracerProvider(
    resource=Resource({ResourceAttributes.PROJECT_NAME: "my-app"}),
    config=TraceConfig(hide_input_images=True),  # optional; also read from env
)
tracer_provider.add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter("http://localhost:6006/v1/traces"))
)
tracer = tracer_provider.get_tracer(__name__)  # an OITracer
```

`openinference.instrumentation.TracerProvider` is a drop-in for the SDK provider that hands
out `OITracer` instances and raises span limits. Phoenix users can call
`phoenix.otel.register(project_name=...)` instead, which does the same.

## Auto-instrument a library

```python
from openinference.instrumentation.openai import OpenAIInstrumentor
OpenAIInstrumentor().instrument(tracer_provider=tracer_provider, config=TraceConfig())
```

Call before the library is used. Names available as `openinference-instrumentation-<name>`:
ag2, agent-framework, agentspec, agno, anthropic, autogen, autogen-agentchat, bedrock, beeai,
claude-agent-sdk, cohere, crewai, dspy, google-adk, google-genai, groq, guardrails, haystack,
instructor, langchain, litellm, llama-index, mcp, mistralai, ollama, openai, openai-agents,
openlit, openllmetry, pipecat, portkey, promptflow, pydantic-ai, smolagents, strands-agents,
together, vertexai. Combine a framework instrumentor with the model-provider instrumentor
for full coverage. `uninstrument()` reverses.

## Trace your own code

Decorators work on sync, async, and generator functions. Input comes from the bound
arguments and output from the return value, JSON-encoded when not a string.

```python
@tracer.agent  # also .chain .tool .llm .retriever .reranker .guardrail .evaluator
def run(question: str) -> str: ...

@tracer.tool  # tool.description and tool.parameters inferred from docstring and signature
def get_weather(city: str) -> str: ...

@tracer.llm(process_input=to_llm_input_attrs, process_output=to_llm_output_attrs)
def call_model(messages: list[dict]) -> Any: ...
```

Manual span:

```python
with tracer.start_as_current_span("rerank", openinference_span_kind="reranker") as span:
    span.set_input(query)  # also set_output(value), set_tool(name=..., parameters=...)
    span.set_attributes(get_reranker_attributes(query=query, output_documents=docs))
```

Attribute helpers in `openinference.instrumentation` produce correctly flattened keys:
`get_llm_attributes(provider=, system=, model_name=, invocation_parameters=, input_messages=,
output_messages=, token_count=, tools=)`, `get_retriever_attributes(documents=)`,
`get_embedding_attributes(model_name=, embeddings=)`, `get_tool_attributes(name=,
description=, parameters=)`, `get_input_attributes(value)`, `get_output_attributes(value)`.
Inputs are the TypedDicts `Message`, `ToolCall`, `TokenCount`, `Tool`, `Document`, `Embedding`.

## Context attributes

```python
from openinference.instrumentation import using_session, using_user, using_metadata, using_tags

with using_session("sess-1"), using_user("u-1"), using_metadata({"tenant": "acme"}):
    run(question)  # every span inside carries session.id, user.id, metadata
```

`using_tags`, `using_prompt_template`, and `using_attributes` (all at once) follow the same
shape. Each also works as a decorator; stack them above `@tracer.*` so the attributes attach
before the span starts. They live on the OTel Context, so new threads or tasks need the
context copied over.

## Suppress tracing

`with suppress_tracing(): ...` from `openinference.instrumentation` around evaluations,
health checks, and internal calls.

## Writing an instrumentor in this repo

Follow `python/DEVELOPMENT.md`. Subclass `BaseInstrumentor`. In `_instrument`, build
`OITracer(trace_api.get_tracer(__name__, __version__, tracer_provider), config=config)` and
patch with `wrapt.wrap_function_wrapper`. In each wrapper, return early when
`context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY)` is set, and add
`get_attributes_from_context()` to every span. Restore the originals in `_uninstrument`.
Tests must cover suppression, context attributes, and TraceConfig masking. Add a tox env in
`python/tox.ini`. Review with the `python-code-reviewer` skill.
