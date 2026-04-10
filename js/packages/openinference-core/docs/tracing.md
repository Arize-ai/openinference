# Tracing Functions and Methods

## withSpan

Wraps any function (sync or async) with OpenTelemetry tracing. Each call creates
a span that records inputs, outputs, errors, and timing.

### Signature

```typescript
function withSpan<Fn extends (...args: any[]) => any>(
  fn: Fn,
  options?: SpanTraceOptions<Fn>,
): Fn;

interface SpanTraceOptions<Fn> {
  name?: string;                           // Span name (defaults to fn.name)
  tracer?: Tracer;                         // OTel Tracer or OITracer instance
  openTelemetrySpanKind?: SpanKind;        // OTel span kind (default: INTERNAL)
  kind?: OpenInferenceSpanKind | `${OpenInferenceSpanKind}`;   // OI span kind (default: CHAIN)
  processInput?: (...args: Parameters<Fn>) => Attributes;
  processOutput?: (result: Awaited<ReturnType<Fn>>) => Attributes;
  attributes?: Attributes;                 // Static attributes added to every span
}
```

### Basic Usage

```typescript
import { withSpan } from "@arizeai/openinference-core";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

// Async function
const fetchAnswer = withSpan(
  async (question: string) => {
    const response = await callLLM(question);
    return response.text;
  },
  { name: "fetch-answer", kind: OpenInferenceSpanKind.LLM },
);

// Sync function
const transform = withSpan(
  (data: string) => data.toUpperCase(),
  { name: "transform" },
);
```

The wrapped function preserves the calling context, so methods keep their
receiver when the traced wrapper is invoked as a method or via `.call()` /
`.apply()`.
Detached method references still need an explicit `.bind(instance)` if you want
to call them without a receiver, because JavaScript does not retain the original
object once a method is extracted.

Synchronous throws and rejected promises are both recorded on the span, mark the
span status as `ERROR`, end the span, and then re-throw the original error.

The `kind` accepts either the enum value (e.g., `OpenInferenceSpanKind.LLM`) or
its string equivalent (e.g., `"LLM"`). Only the uppercase enum value strings are
valid -- `"llm"` or `"custom"` will be rejected by the type system.

### Span Naming

- If `name` is provided in options, it is used
- Otherwise, `fn.name` is used (the function's declared name)
- Arrow functions have empty names -- always provide `name` for arrow functions

### Custom Input/Output Processors

By default, `withSpan` uses `defaultProcessInput` and `defaultProcessOutput` which
auto-detect strings vs objects and set `input.value`/`output.value` with the
appropriate MIME type. Override these for domain-specific attributes:

```typescript
import {
  getInputAttributes,
  getRetrieverAttributes,
  withSpan,
} from "@arizeai/openinference-core";

const retriever = withSpan(
  async (query: string) => {
    const docs = await vectorSearch(query);
    return docs;
  },
  {
    name: "vector-search",
    kind: "RETRIEVER",
    processInput: (query) => getInputAttributes(query),
    processOutput: (docs) =>
      getRetrieverAttributes({
        documents: docs.map((doc) => ({
          content: doc.text,
          id: doc.id,
          score: doc.score,
        })),
      }),
  },
);
```

### Accessing the Active Span

Inside a wrapped function, use `trace.getActiveSpan()` to add extra attributes
mid-execution:

```typescript
import { trace } from "@opentelemetry/api";
import { getLLMAttributes, withSpan } from "@arizeai/openinference-core";

const chat = withSpan(
  async (prompt: string) => {
    const result = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: prompt }],
    });

    // Add LLM-specific attributes after the call completes
    const span = trace.getActiveSpan();
    span?.setAttributes(
      getLLMAttributes({
        provider: "openai",
        modelName: "gpt-4o",
        inputMessages: [{ role: "user", content: prompt }],
        outputMessages: [{ role: "assistant", content: result.choices[0].message.content ?? "" }],
        tokenCount: {
          prompt: result.usage?.prompt_tokens,
          completion: result.usage?.completion_tokens,
          total: result.usage?.total_tokens,
        },
      }),
    );

    return result.choices[0].message.content;
  },
  { name: "chat", kind: "LLM" },
);
```

### Base Attributes

Use the `attributes` option to add static metadata to every span:

```typescript
const traced = withSpan(myFn, {
  name: "my-operation",
  attributes: {
    "service.version": "2.1.0",
    "deployment.environment": "production",
  },
});
```

## traceChain, traceAgent, traceTool

Convenience wrappers that call `withSpan` with the `kind` pre-set. Their options
type is `Omit<SpanTraceOptions, "kind">`.

```typescript
import { traceAgent, traceChain, traceTool } from "@arizeai/openinference-core";

const chain = traceChain(myPipeline, { name: "rag-chain" });     // kind = CHAIN
const agent = traceAgent(myOrchestrator, { name: "qa-agent" });  // kind = AGENT
const tool  = traceTool(myApiCall, { name: "weather-lookup" });   // kind = TOOL
```

### When to Use Which

| Wrapper | Span Kind | Use For |
|---------|-----------|---------|
| `traceChain` | CHAIN | Multi-step workflows, pipelines, sequential processing |
| `traceAgent` | AGENT | Autonomous agents, decision-making loops, orchestrators |
| `traceTool` | TOOL | External API calls, database queries, calculators |
| `withSpan` | (any) | When you need RETRIEVER, LLM, EMBEDDING, RERANKER, GUARDRAIL, EVALUATOR, or any other kind |

### Nested Tracing Example

Wrapped functions automatically create parent-child span relationships:

```typescript
import {
  getInputAttributes,
  getRetrieverAttributes,
  traceAgent,
  traceTool,
  withSpan,
} from "@arizeai/openinference-core";

const retrieve = withSpan(
  async (query: string) => {
    return await vectorDB.search(query);
  },
  {
    name: "vector-search",
    kind: "RETRIEVER",
    processInput: (query) => getInputAttributes(query),
    processOutput: (docs) =>
      getRetrieverAttributes({ documents: docs }),
  },
);

const summarize = withSpan(
  async (context: string, question: string) => {
    return await llm.complete(`${context}\n\nQ: ${question}`);
  },
  { name: "summarize", kind: "LLM" },
);

const lookupWeather = traceTool(
  async (city: string) => {
    const res = await fetch(`https://api.weather.com/${city}`);
    return res.json();
  },
  { name: "weather-lookup" },
);

// The agent orchestrates -- its span parents the child spans
const agent = traceAgent(
  async (question: string) => {
    if (question.includes("weather")) {
      return await lookupWeather("seattle");
    }
    const docs = await retrieve(question);
    const context = docs.map((d) => d.content).join("\n");
    return await summarize(context, question);
  },
  { name: "qa-agent" },
);
```

This produces a trace tree:

```
AGENT: "qa-agent"
  |-- RETRIEVER: "vector-search"
  |-- LLM: "summarize"
```

## @observe Decorator

Decorator factory for tracing class methods. Uses TypeScript 5+ standard
decorators (not legacy/experimental decorators).

### Signature

```typescript
function observe<Fn extends (...args: any[]) => any>(
  options?: SpanTraceOptions,
): (originalMethod: Fn, ctx: ClassMethodDecoratorContext) => Fn;
```

### Requirements

- TypeScript 5+ standard decorators
- Must be applied to class methods (not standalone functions)
- Preserves `this` context -- safe to use with class instance state

### Usage

```typescript
import { observe } from "@arizeai/openinference-core";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

class RAGService {
  private db: VectorDB;

  @observe({ name: "retrieve", kind: OpenInferenceSpanKind.RETRIEVER })
  async retrieve(query: string) {
    return await this.db.search(query);
  }

  @observe({ kind: OpenInferenceSpanKind.LLM })
  async generate(prompt: string, context: string[]) {
    return await this.llm.complete(prompt, { context });
  }

  @observe({ kind: OpenInferenceSpanKind.CHAIN })
  async answer(question: string) {
    const docs = await this.retrieve(question);
    const context = docs.map((d) => d.content);
    return await this.generate(question, context);
  }
}
```

### Combining with Attribute Helpers

Use `trace.getActiveSpan()` inside a decorated method to add structured attributes:

```typescript
import { trace } from "@opentelemetry/api";
import { getEmbeddingAttributes, observe } from "@arizeai/openinference-core";

class EmbeddingService {
  @observe({ kind: "EMBEDDING" })
  async embed(texts: string[]) {
    const result = await this.model.embed(texts);

    trace.getActiveSpan()?.setAttributes(
      getEmbeddingAttributes({
        modelName: "text-embedding-3-small",
        embeddings: result.map((vector, i) => ({
          text: texts[i],
          vector,
        })),
      }),
    );

    return result;
  }
}
```

## Providing a Custom Tracer

By default, `withSpan` and `@observe` use the global tracer provider. To use a
specific tracer (e.g., with data masking), pass the `tracer` option:

```typescript
import { trace } from "@opentelemetry/api";
import { OITracer, withSpan } from "@arizeai/openinference-core";

// Option 1: Pass a raw OTel tracer (auto-wrapped in OITracer)
const traced1 = withSpan(myFn, {
  tracer: trace.getTracer("my-service"),
});

// Option 2: Pass an OITracer with masking config
const oiTracer = new OITracer({
  tracer: trace.getTracer("my-service"),
  traceConfig: { hideInputs: true },
});
const traced2 = withSpan(myFn, { tracer: oiTracer });
```

See [trace-config-and-masking.md](./trace-config-and-masking.md) for full masking
configuration details.
