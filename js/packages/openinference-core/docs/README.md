# @arizeai/openinference-core Documentation

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

## Minimal Setup

```typescript
import {
  OpenInferenceSpanKind,
  SEMRESATTRS_PROJECT_NAME,
} from "@arizeai/openinference-semantic-conventions";
import {
  ConsoleSpanExporter,
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { withSpan } from "@arizeai/openinference-core";

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "my-project",
  }),
  spanProcessors: [new SimpleSpanProcessor(new ConsoleSpanExporter())],
});
provider.register();

const greet = withSpan(
  async (name: string) => `Hello, ${name}!`,
  { name: "greet", kind: OpenInferenceSpanKind.CHAIN },
);

async function main() {
  await greet("world");
}

main();
```

## Documentation Guide

| Document | When to Read It |
|----------|----------------|
| [tracing.md](./tracing.md) | Wrapping functions or class methods with tracing (`withSpan`, `traceChain`, `@observe`) |
| [context-attributes.md](./context-attributes.md) | Propagating session, user, metadata, or tags across spans |
| [attribute-helpers.md](./attribute-helpers.md) | Adding LLM, embedding, retriever, or tool attributes to spans |
| [trace-config-and-masking.md](./trace-config-and-masking.md) | Hiding sensitive data from traces with `OITracer` |

## Docs and Source Code in node_modules

Once you've installed the openinference-core package, you already have the full
openinference-core documentation and source code available locally inside
node_modules. Your coding agent can read these directly -- no internet access
required.

```
node_modules/@arizeai/openinference-core/src/              # Full source code organized by module
node_modules/@arizeai/openinference-core/docs/             # Official documentation with examples
```

This means your agent can look up accurate API signatures, implementations, and
usage examples directly from the installed package -- ensuring it always uses the
version of the SDK that's actually installed in your project.

---

## Core Concepts

### OpenTelemetry Basics

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

### OpenInference Span Kinds

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

#### How Span Kinds Compose

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

### The Tracing Stack

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

### Data Flow

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

If the function throws or returns a rejected promise, step 5 is skipped.
Instead, the exception is recorded on the span, the status is set to ERROR, the
span is ended, and the error is re-thrown.

### Semantic Conventions

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

---

## All Exports at a Glance

**Function Wrappers**
- `withSpan(fn, options?)` -- wrap any function with a traced span
- `traceChain(fn, options?)` -- wrap with CHAIN span kind
- `traceAgent(fn, options?)` -- wrap with AGENT span kind
- `traceTool(fn, options?)` -- wrap with TOOL span kind

**Decorator**
- `observe(options?)` -- class method decorator for tracing

**Context Attributes**
- `setSession(context, { sessionId })` / `getSession` / `clearSession`
- `setUser(context, { userId })` / `getUser` / `clearUser`
- `setMetadata(context, metadata)` / `getMetadata` / `clearMetadata`
- `setTags(context, tags)` / `getTags` / `clearTags`
- `setPromptTemplate(context, { template, variables?, version? })` / `getPromptTemplate` / `clearPromptTemplate`
- `setAttributes(context, attributes)` / `getAttributes` / `clearAttributes`
- `getAttributesFromContext(context)` -- extract all propagated attributes for a span

**Attribute Helpers**
- `getLLMAttributes({ provider?, modelName?, inputMessages?, outputMessages?, tokenCount?, tools?, invocationParameters? })`
- `getEmbeddingAttributes({ modelName?, embeddings? })`
- `getRetrieverAttributes({ documents })`
- `getDocumentAttributes(document, documentIndex, keyPrefix)` -- single document with custom key prefix
- `getToolAttributes({ name, description?, parameters })`
- `getMetadataAttributes(metadata)`
- `getInputAttributes(input)` / `getOutputAttributes(output)`
- `defaultProcessInput(...args)` / `defaultProcessOutput(result)`

**Trace Config & Masking**
- `OITracer` -- tracer wrapper with context propagation and data masking
- `OISpan` -- span wrapper that applies masking rules
- `generateTraceConfig(options?)` -- merge options + env vars + defaults
- `wrapTracer(tracer)` / `getTracer(name?)` -- tracer utilities

**Utilities**
- `withSafety({ fn, onError? })` -- wrap function with try-catch, returns null on error
- `safelyJSONStringify(value)` / `safelyJSONParse(value)` -- guarded JSON operations

## Source Code Map

```
src/
  index.ts                          # Main entry point (re-exports everything)
  helpers/
    withSpan.ts                     # withSpan implementation
    wrappers.ts                     # traceChain, traceAgent, traceTool
    decorators.ts                   # @observe decorator
    attributeHelpers.ts             # getLLMAttributes, getEmbeddingAttributes, etc.
    tracerHelpers.ts                # getTracer, wrapTracer
    types.ts                        # SpanTraceOptions, SpanInput/Output, Message, TokenCount, etc.
  trace/
    contextAttributes.ts            # setSession, setUser, setMetadata, etc.
    types.ts                        # Session, User, Metadata, PromptTemplate, Tags
    trace-config/
      OITracer.ts                   # OITracer class
      OISpan.ts                     # OISpan class (masking-aware span wrapper)
      traceConfig.ts                # generateTraceConfig
      maskingRules.ts               # Masking rule definitions
      types.ts                      # TraceConfigOptions, TraceConfig, MaskingRule
      constants.ts                  # Environment variable names and defaults
  utils/
    index.ts                        # withSafety, safelyJSONStringify, safelyJSONParse
    typeUtils.ts                    # isPromise, isAttributes, etc.
    types.ts                        # GenericFunction, SafeFunction
```
