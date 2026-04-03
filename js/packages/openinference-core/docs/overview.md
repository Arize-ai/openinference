# Core Concepts

## What is OpenInference?

OpenInference is an open standard for tracing LLM applications, built on top of
[OpenTelemetry](https://opentelemetry.io/). It defines semantic conventions --
standardized attribute names and span structures -- that describe LLM-specific
operations like model inference, document retrieval, embedding generation, tool
usage, and agent orchestration.

`@arizeai/openinference-core` is the JavaScript/TypeScript foundation. It provides
the tracing primitives, context propagation, attribute helpers, and data masking
that all OpenInference JS instrumentations build on. You can also use it directly
to trace your own application code.

## OpenTelemetry Basics

OpenInference extends OpenTelemetry, so a few OTel concepts are essential:

- **Tracer Provider** -- creates and manages tracers. You configure it once at
  startup with a span exporter (where trace data goes) and register it globally.
- **Tracer** -- creates spans. OpenInference wraps OTel tracers with `OITracer`
  to add context propagation and data masking.
- **Span** -- represents a single operation with a name, start/end time, status,
  and key-value attributes. Spans nest to form traces.
- **Context** -- carries request-scoped data (like session ID or user ID) across
  function boundaries. OpenInference uses context to propagate attributes to all
  child spans automatically.
- **Attributes** -- key-value pairs attached to spans. OpenInference defines
  semantic conventions for attribute keys (e.g., `input.value`, `llm.model_name`).
- **Exporter** -- sends completed spans to a backend. Common choices:
  `ConsoleSpanExporter` (stdout), `OTLPTraceExporter` (to Phoenix, Jaeger, etc.).

```typescript
import { NodeTracerProvider, SimpleSpanProcessor, ConsoleSpanExporter } from "@opentelemetry/sdk-trace-node";

// Set up once at application startup
const provider = new NodeTracerProvider({
  spanProcessors: [new SimpleSpanProcessor(new ConsoleSpanExporter())],
});
provider.register(); // Makes this provider the global default
```

## OpenInference Span Kinds

Every OpenInference span has a `kind` that describes its role in an LLM workflow.
These are domain-level classifications (separate from OTel's `SpanKind`):

| Span Kind | Purpose | Example |
|-----------|---------|---------|
| **CHAIN** | A multi-step workflow or pipeline | RAG pipeline, data processing sequence |
| **LLM** | A language model inference call | ChatGPT completion, Claude message |
| **RETRIEVER** | A document/knowledge retrieval operation | Vector search, keyword search |
| **EMBEDDING** | An embedding generation operation | text-embedding-ada-002 call |
| **AGENT** | An autonomous decision-making entity | ReAct agent, tool-using agent |
| **TOOL** | An external tool or function call | Weather API, calculator, database query |
| **RERANKER** | A document reranking operation | Cross-encoder reranker, Cohere rerank |
| **GUARDRAIL** | An input/output safety check | Content filter, PII detector, toxicity check |
| **EVALUATOR** | A quality or correctness evaluation | LLM-as-judge, relevance scorer |

### How Span Kinds Compose

Span kinds form parent-child relationships in a trace tree. A typical RAG agent
might produce this trace structure:

```
AGENT: "qa-agent"
  |-- RETRIEVER: "vector-search"
  |     |-- EMBEDDING: "embed-query"
  |-- LLM: "generate-answer"
  |-- TOOL: "format-citations"
```

- An **AGENT** typically parents other spans as it orchestrates work
- A **CHAIN** groups sequential steps without autonomous decision-making
- **LLM**, **RETRIEVER**, **EMBEDDING**, **TOOL**, **RERANKER**, **GUARDRAIL**, and **EVALUATOR** are usually leaf spans

## The Tracing Stack

OpenInference layers on top of OpenTelemetry:

```
+----------------------------------------------------------+
|  Your Application Code                                   |
|  (functions, classes, handlers)                           |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
|  Tracing API                                             |
|  withSpan() / traceChain() / traceAgent() / @observe()   |
|  Wraps your functions, creates spans automatically        |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
|  Attribute Helpers                                       |
|  getLLMAttributes() / getRetrieverAttributes() / ...      |
|  Converts domain objects into OTel attributes             |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
|  OITracer + OISpan                                       |
|  Wraps OTel Tracer/Span to add:                          |
|  - Context attribute propagation (session, user, etc.)    |
|  - Data masking (hide inputs, redact PII)                 |
+----------------------------------------------------------+
        |
        v
+----------------------------------------------------------+
|  OpenTelemetry SDK                                       |
|  NodeTracerProvider -> SpanProcessor -> Exporter          |
|  Delivers spans to your observability backend             |
+----------------------------------------------------------+
```

## Data Flow

When you call a function wrapped with `withSpan`, here is what happens:

1. **Span created** -- OITracer creates a new span via the OTel tracer
2. **Context attributes injected** -- session ID, user ID, metadata, tags, and
   any other attributes set on the current context are merged into the span
3. **Input processed** -- function arguments are converted to span attributes
   using `processInput` (or `defaultProcessInput` if not specified)
4. **Function executes** -- your original function runs normally
5. **Output processed** -- the return value is converted to span attributes
   using `processOutput` (or `defaultProcessOutput` if not specified)
6. **Masking applied** -- if `OITracer` was configured with a `traceConfig`,
   sensitive attributes are redacted or removed before being written
7. **Span ended** -- the span is finalized with a status (OK or ERROR) and
   handed to the span processor for export
8. **Exported** -- the span processor sends the span to your configured exporter

If the function throws, step 5 is skipped. Instead, the exception is recorded on
the span, the status is set to ERROR, and the error is re-thrown.

## Semantic Conventions

The `@arizeai/openinference-semantic-conventions` package defines the attribute
keys used by OpenInference (e.g., `input.value`, `output.value`,
`llm.model_name`, `retrieval.documents.0.document.content`).

You rarely need to use these constants directly. The attribute helpers
(`getLLMAttributes`, `getRetrieverAttributes`, etc.) abstract over them, producing
correctly-keyed attributes from simple objects. For example:

```typescript
import { getLLMAttributes } from "@arizeai/openinference-core";

// Instead of manually setting:
//   span.setAttribute("llm.model_name", "gpt-4")
//   span.setAttribute("llm.input_messages.0.message.role", "user")
//   span.setAttribute("llm.input_messages.0.message.content", "Hello")
//   ...

// Use the helper:
span.setAttributes(
  getLLMAttributes({
    modelName: "gpt-4",
    inputMessages: [{ role: "user", content: "Hello" }],
  }),
);
```

See [attribute-helpers.md](./attribute-helpers.md) for the complete reference.
