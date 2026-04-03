# @arizeai/openinference-core Documentation

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

await greet("world");
```

## Documentation Guide

| Document | When to Read It |
|----------|----------------|
| [overview.md](./overview.md) | Understanding core concepts, span kinds, and how the tracing stack fits together |
| [tracing.md](./tracing.md) | Wrapping functions or class methods with tracing (`withSpan`, `traceChain`, `@observe`) |
| [context-attributes.md](./context-attributes.md) | Propagating session, user, metadata, or tags across spans |
| [attribute-helpers.md](./attribute-helpers.md) | Adding LLM, embedding, retriever, or tool attributes to spans |
| [trace-config-and-masking.md](./trace-config-and-masking.md) | Hiding sensitive data from traces with `OITracer` |

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
