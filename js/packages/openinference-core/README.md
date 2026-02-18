# OpenInference Core

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-core.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-core)

`@arizeai/openinference-core` is the shared tracing foundation for OpenInference JS packages. It provides:

- context attribute propagation (`session.id`, `user.id`, metadata, tags, prompt template)
- span wrappers (`withSpan`, `traceChain`, `traceAgent`, `traceTool`)
- method decorator tracing (`@observe`)
- helpers for LLM/retrieval/embedding/tool attributes
- optional sensitive-data masking through `OITracer` trace config

## Installation

```bash
npm install @arizeai/openinference-core

# Only needed if you want to run the Quick Start example in this README:
npm install @arizeai/openinference-semantic-conventions @opentelemetry/sdk-trace-node @opentelemetry/resources
```

## Quick Start (Runnable)

This example exports spans to stdout and sets an OpenInference project name.

```typescript
import {
  OpenInferenceSpanKind,
  SEMRESATTRS_PROJECT_NAME,
} from "@arizeai/openinference-semantic-conventions";
import { ConsoleSpanExporter, NodeTracerProvider, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";
import { resourceFromAttributes } from "@opentelemetry/resources";

import { withSpan } from "@arizeai/openinference-core";

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "openinference-core-demo",
  }),
  spanProcessors: [new SimpleSpanProcessor(new ConsoleSpanExporter())],
});

provider.register();

const answerQuestion = withSpan(
  async (question: string) => {
    return `Answer: ${question}`;
  },
  {
    name: "answer-question",
    kind: OpenInferenceSpanKind.CHAIN,
  },
);

async function main() {
  const answer = await answerQuestion("What is OpenInference?");
  console.log(answer);
}

void main();
```

## Context Attributes

Each setter returns a new OpenTelemetry context. Compose them to propagate request-level attributes:

- `setSession(context, { sessionId })`
- `setUser(context, { userId })`
- `setMetadata(context, metadataObject)`
- `setTags(context, string[])`
- `setPromptTemplate(context, { template, variables?, version? })`
- `setAttributes(context, attributes)`

```typescript
import { context } from "@opentelemetry/api";
import {
  setAttributes,
  setMetadata,
  setPromptTemplate,
  setSession,
  setTags,
  setUser,
} from "@arizeai/openinference-core";

const enrichedContext = setAttributes(
  setPromptTemplate(
    setTags(
      setMetadata(
        setUser(setSession(context.active(), { sessionId: "sess-42" }), {
          userId: "user-7",
        }),
        { tenant: "acme", environment: "prod" },
      ),
      ["support", "priority-high"],
    ),
    {
      template: "Answer using docs about {topic}",
      variables: { topic: "billing" },
      version: "v3",
    },
  ),
  {
    "app.request_id": "req-123",
  },
);

context.with(enrichedContext, async () => {
  // spans started in this context by openinference-core wrappers
  // include these propagated attributes automatically
});
```

If you create spans manually with a plain OpenTelemetry tracer, apply propagated attributes explicitly:

```typescript
import { context, trace } from "@opentelemetry/api";
import { getAttributesFromContext } from "@arizeai/openinference-core";

const tracer = trace.getTracer("manual-tracer");
const span = tracer.startSpan("manual-span");
span.setAttributes(getAttributesFromContext(context.active()));
span.end();
```

## Function Wrappers

### `withSpan`

```typescript
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";
import { withSpan } from "@arizeai/openinference-core";

const retrieve = withSpan(
  async (query: string) => {
    return [`Document for ${query}`];
  },
  {
    name: "retrieve-documents",
    kind: OpenInferenceSpanKind.RETRIEVER,
  },
);
```

### `traceChain`, `traceAgent`, `traceTool`

These wrappers call `withSpan` and set `kind` automatically.

```typescript
import { traceAgent, traceChain, traceTool } from "@arizeai/openinference-core";

const tracedChain = traceChain(async (q: string) => `chain result: ${q}`, {
  name: "rag-chain",
});

const tracedTool = traceTool(async (city: string) => ({ temp: 72, city }), {
  name: "weather-tool",
});

const tracedAgent = traceAgent(async (q: string) => {
  const toolResult = await tracedTool("seattle");
  return tracedChain(`${q} (${toolResult.temp}F)`);
}, { name: "qa-agent" });
```

### Custom input/output processors

```typescript
import {
  getInputAttributes,
  getRetrieverAttributes,
  withSpan,
} from "@arizeai/openinference-core";

const retriever = withSpan(
  async (query: string) => [`Doc A for ${query}`, `Doc B for ${query}`],
  {
    name: "retriever",
    kind: "RETRIEVER",
    processInput: (query) => getInputAttributes(query),
    processOutput: (documents) =>
      getRetrieverAttributes({
        documents: documents.map((content, i) => ({
          id: `doc-${i}`,
          content,
        })),
      }),
  },
);
```

## Decorators (`@observe`)

`observe` wraps class methods with tracing and preserves method `this` context.
Use TypeScript 5+ standard decorators when applying `@observe`.

```typescript
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";
import { observe } from "@arizeai/openinference-core";

class ChatService {
  @observe({ kind: OpenInferenceSpanKind.CHAIN })
  async runWorkflow(message: string) {
    return `processed: ${message}`;
  }

  @observe({ name: "llm-call", kind: OpenInferenceSpanKind.LLM })
  async callModel(prompt: string) {
    return `model output for: ${prompt}`;
  }
}
```

## Attribute Helper APIs

Use these helpers to generate OpenInference-compatible attributes and attach them to spans:

- `getLLMAttributes({ provider, modelName, inputMessages, outputMessages, tokenCount, tools, ... })`
- `getEmbeddingAttributes({ modelName, embeddings })`
- `getRetrieverAttributes({ documents })`
- `getToolAttributes({ name, description?, parameters })`
- `getMetadataAttributes(metadataObject)`
- `getInputAttributes(input)` / `getOutputAttributes(output)`
- `defaultProcessInput(...args)` / `defaultProcessOutput(result)`

Example:

```typescript
import { trace } from "@opentelemetry/api";
import { getLLMAttributes } from "@arizeai/openinference-core";

const tracer = trace.getTracer("llm-service");

tracer.startActiveSpan("llm-inference", (span) => {
  span.setAttributes(
    getLLMAttributes({
      provider: "openai",
      modelName: "gpt-4o-mini",
      inputMessages: [{ role: "user", content: "What is OpenInference?" }],
      outputMessages: [{ role: "assistant", content: "OpenInference is..." }],
      tokenCount: { prompt: 12, completion: 44, total: 56 },
      invocationParameters: { temperature: 0.2 },
    }),
  );
  span.end();
});
```

## Trace Config and Redaction (`OITracer`)

`OITracer` wraps an OpenTelemetry tracer and can redact or drop sensitive attributes before writing spans:

```typescript
import { trace } from "@opentelemetry/api";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";
import { OITracer, withSpan } from "@arizeai/openinference-core";

const tracer = new OITracer({
  tracer: trace.getTracer("my-service"),
  traceConfig: {
    hideInputs: true,
    hideOutputText: true,
    hideEmbeddingVectors: true,
    base64ImageMaxLength: 8_000,
  },
});

const traced = withSpan(
  async (prompt: string) => `model response for ${prompt}`,
  {
    tracer,
    kind: OpenInferenceSpanKind.LLM,
    name: "safe-llm-call",
  },
);
```

You can also configure masking with environment variables:

- `OPENINFERENCE_HIDE_INPUTS`
- `OPENINFERENCE_HIDE_OUTPUTS`
- `OPENINFERENCE_HIDE_INPUT_MESSAGES`
- `OPENINFERENCE_HIDE_OUTPUT_MESSAGES`
- `OPENINFERENCE_HIDE_INPUT_IMAGES`
- `OPENINFERENCE_HIDE_INPUT_TEXT`
- `OPENINFERENCE_HIDE_OUTPUT_TEXT`
- `OPENINFERENCE_HIDE_EMBEDDING_VECTORS`
- `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH`
- `OPENINFERENCE_HIDE_PROMPTS`

## Utility Helpers

- `withSafety({ fn, onError? })`: wraps a function and returns `null` on error
- `safelyJSONStringify(value)` / `safelyJSONParse(value)`: guarded JSON operations

## Documentation

- API reference: [@arizeai/openinference-core](https://arize-ai.github.io/openinference/js/modules/_arizeai_openinference-core.html)
- OpenInference JS docs: [openinference/js](https://arize-ai.github.io/openinference/js/)
- Source code: [js/packages/openinference-core](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-core)
