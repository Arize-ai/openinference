---
name: openinference-instrument-js
description: >
  Instrument JavaScript and TypeScript LLM, agent, and RAG code with OpenInference on
  OpenTelemetry: set up a NodeTracerProvider with a project name, register
  auto-instrumentations (OpenAI, Anthropic, LangChain, Bedrock, Google GenAI, MCP), use the
  Vercel AI SDK span processor and TanStack AI middleware, trace your own functions with withSpan,
  traceChain, traceAgent, traceTool, or the observe decorator, build attributes with helpers,
  attach context attributes (setSession, setUser, setMetadata, setTags), mask data with
  OITracer trace config, pause tracing with suppressTracing, and follow the checklist for
  writing a new instrumentation in this repo. Use for any Node or TypeScript tracing question
  involving OpenInference, Phoenix, or Arize.
---

# OpenInference for JavaScript and TypeScript

Concepts live in the `openinference` skill; read it first. Packages:
`@arizeai/openinference-core` (helpers), `@arizeai/openinference-semantic-conventions`
(attribute constants), `@arizeai/openinference-instrumentation-<lib>` for openai, anthropic,
langchain, bedrock, bedrock-agent-runtime, google-genai, mcp, openai-agents, claude-agent-sdk,
beeai. The Vercel AI SDK emits its own OTel spans, so `@arizeai/openinference-vercel` ships
`OpenInferenceSimpleSpanProcessor` and `OpenInferenceBatchSpanProcessor` that rewrite them instead of patching modules;
`@arizeai/openinference-tanstack-ai` exports an `openInferenceMiddleware` for TanStack AI.

## Setup

```ts
import { NodeTracerProvider, BatchSpanProcessor } from "@opentelemetry/sdk-trace-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";

const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({ [SEMRESATTRS_PROJECT_NAME]: "my-app" }),
  spanProcessors: [
    new BatchSpanProcessor(new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" })),
  ],
});
provider.register(); // also installs the async context manager
registerInstrumentations({
  instrumentations: [new OpenAIInstrumentation({ traceConfig: { hideInputImages: true } })],
});
```

Load this file before the instrumented library, for example with
`node --import ./instrumentation.js app.js`. Under ESM or bundlers, module patching can miss;
import the library and call `instrumentation.manuallyInstrument(module)` instead.

## Trace your own code

```ts
import { withSpan, traceAgent, traceTool, getInputAttributes, getRetrieverAttributes } from "@arizeai/openinference-core";
import { OpenInferenceSpanKind } from "@arizeai/openinference-semantic-conventions";

const run = traceAgent(async (question: string) => { /* ... */ }, { name: "run" });
const getWeather = traceTool(async (city: string) => { /* ... */ }, { name: "get_weather" });
const retrieve = withSpan(retrieveDocs, {
  kind: OpenInferenceSpanKind.RETRIEVER,
  name: "retrieve",
  processInput: (query) => getInputAttributes(query),
  processOutput: (docs) => getRetrieverAttributes({ documents: docs }),
});
```

Also `traceChain`, `traceLLM`, `traceRetriever`, `traceReranker`, `traceEmbedding`,
`traceGuardrail`, `traceEvaluator`, `tracePrompt`, and `@observe({ kind })` on class methods
(TypeScript 5 standard decorators). Wrappers preserve `this`, await promises, and record
errors. Without processors, input and output are JSON-stringified. Pass `tracer` to control
masking; `getTracer(name)` and `wrapTracer(tracer)` return an `OITracer`.

Attribute helpers produce correctly flattened keys: `getLLMAttributes({ provider, system,
modelName, invocationParameters, inputMessages, outputMessages, tokenCount, tools })`,
`getInputAttributes`, `getOutputAttributes`, `getToolAttributes({ name, description, parameters })`,
`getRetrieverAttributes({ documents })`, `getEmbeddingAttributes`, `getDocumentAttributes`.

## Context attributes

```ts
import { context } from "@opentelemetry/api";
import { setSession, setUser, setMetadata } from "@arizeai/openinference-core";

let ctx = setSession(context.active(), { sessionId: "sess-1" });
ctx = setUser(ctx, { userId: "u-1" });
ctx = setMetadata(ctx, { tenant: "acme" });
await context.with(ctx, () => run(question)); // every span inside carries them
```

`setTags`, `setPromptTemplate`, and `setAttributes` follow the same shape. For spans made with
a raw tracer, call `span.setAttributes(getAttributesFromContext(context.active()))`.

## Suppress tracing

`context.with(suppressTracing(context.active()), () => ...)` with `suppressTracing` from
`@opentelemetry/core`.

## Trace config

`new OITracer({ tracer, traceConfig })` and every instrumentation constructor accept the same
`traceConfig`: `hideInputs`, `hideOutputs`, `hideInputMessages`, `hideOutputMessages`,
`hideInputText`, `hideOutputText`, `hideInputImages`, `hideLLMTools`, `hideEmbeddingVectors`,
`hidePrompts`, `base64ImageMaxLength`. Unset keys fall back to `OPENINFERENCE_*` env vars,
then defaults.

## Writing an instrumentation in this repo

Follow `js/CLAUDE.md`. Extend `InstrumentationBase`, wrap `this.tracer` in `OITracer`, return
early when `isTracingSuppressed(context.active())`, add `getAttributesFromContext` to every
span, use SDK types via `import type`, and log with `diag`, never `console`. Tests use Vitest
with manual module mocking and must cover suppression, context attributes, and trace config.
Run `pnpm changeset` before opening the PR.
