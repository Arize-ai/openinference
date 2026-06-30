# Context Attributes

## How Context Propagation Works

OpenInference uses OpenTelemetry's Context API to propagate request-level metadata
(session ID, user ID, metadata, tags, etc.) across function boundaries. When you
set attributes on a context and run code within that context, all spans created by
openinference-core wrappers (`withSpan`, `@observe`, etc.) automatically include
those attributes.

The pattern is:

1. Call `setSession`, `setUser`, `setMetadata`, etc. -- each returns a **new** Context
2. Pass the enriched context to `context.with(enrichedContext, fn)`
3. All spans created inside `fn` inherit the propagated attributes

```typescript
import { context } from "@opentelemetry/api";
import { setSession, setUser } from "@arizeai/openinference-core";

const enriched = setUser(
  setSession(context.active(), { sessionId: "sess-42" }),
  { userId: "user-7" },
);

context.with(enriched, async () => {
  // Every span created here (by withSpan, @observe, etc.)
  // will include session.id="sess-42" and user.id="user-7"
  await myTracedFunction();
});
```

## API Reference

### Session

```typescript
setSession(context: Context, session: { sessionId: string }): Context
getSession(context: Context): { sessionId: string } | undefined
clearSession(context: Context): Context
```

```typescript
import { context } from "@opentelemetry/api";
import { setSession } from "@arizeai/openinference-core";

context.with(
  setSession(context.active(), { sessionId: "sess-42" }),
  () => { /* spans here include session.id */ },
);
```

### User

```typescript
setUser(context: Context, user: { userId: string }): Context
getUser(context: Context): { userId: string } | undefined
clearUser(context: Context): Context
```

```typescript
import { context } from "@opentelemetry/api";
import { setUser } from "@arizeai/openinference-core";

context.with(
  setUser(context.active(), { userId: "user-7" }),
  () => { /* spans here include user.id */ },
);
```

### Metadata

Arbitrary key-value pairs attached as JSON to every span in the context.

```typescript
setMetadata(context: Context, metadata: Record<string, unknown>): Context
getMetadata(context: Context): Record<string, unknown> | undefined
clearMetadata(context: Context): Context
```

```typescript
import { context } from "@opentelemetry/api";
import { setMetadata } from "@arizeai/openinference-core";

context.with(
  setMetadata(context.active(), {
    tenant: "acme",
    environment: "prod",
    requestId: "req-123",
  }),
  () => { /* spans here include metadata */ },
);
```

### Tags

An array of string labels attached to every span in the context.

```typescript
setTags(context: Context, tags: string[]): Context
getTags(context: Context): string[] | undefined
clearTags(context: Context): Context
```

```typescript
import { context } from "@opentelemetry/api";
import { setTags } from "@arizeai/openinference-core";

context.with(
  setTags(context.active(), ["support", "priority-high", "v2"]),
  () => { /* spans here include tag.tags */ },
);
```

### Prompt Template

Attach a prompt template with optional variables and version.

```typescript
setPromptTemplate(context: Context, promptTemplate: {
  template: string;
  variables?: Record<string, unknown>;
  version?: string;
}): Context
getPromptTemplate(context: Context): {
  template?: string;
  variables?: Record<string, unknown>;
  version?: string;
} | undefined
clearPromptTemplate(context: Context): Context
```

```typescript
import { context } from "@opentelemetry/api";
import { setPromptTemplate } from "@arizeai/openinference-core";

context.with(
  setPromptTemplate(context.active(), {
    template: "Answer the question about {topic} using the provided context.",
    variables: { topic: "billing" },
    version: "v3",
  }),
  () => { /* spans here include prompt_template attributes */ },
);
```

### Generic Attributes

Set arbitrary OpenTelemetry attributes that will be propagated to all child spans.

```typescript
setAttributes(context: Context, attributes: Attributes): Context
getAttributes(context: Context): Attributes | undefined
clearAttributes(context: Context): Context
```

```typescript
import { context } from "@opentelemetry/api";
import { setAttributes } from "@arizeai/openinference-core";

context.with(
  setAttributes(context.active(), {
    "app.request_id": "req-123",
    "app.feature_flag": "new-model-enabled",
  }),
  () => { /* spans here include these custom attributes */ },
);
```

## Composing Multiple Context Attributes

Each setter returns a new context. Build up the context step by step:

```typescript
import { context } from "@opentelemetry/api";
import {
  setMetadata,
  setSession,
  setTags,
  setUser,
} from "@arizeai/openinference-core";

let ctx = context.active();
ctx = setSession(ctx, { sessionId: "sess-42" });
ctx = setUser(ctx, { userId: "user-7" });
ctx = setMetadata(ctx, { tenant: "acme", environment: "prod" });
ctx = setTags(ctx, ["support", "priority-high"]);

context.with(ctx, async () => {
  // All spans created here include session, user, metadata, and tags
  await myAgent("How do I update my billing?");
});
```

## getAttributesFromContext

Extracts all OpenInference context attributes as a flat `Attributes` object.
This is used internally by `OITracer` but is also useful when creating spans
with a raw OpenTelemetry tracer:

```typescript
import { context, trace } from "@opentelemetry/api";
import { getAttributesFromContext, setSession } from "@arizeai/openinference-core";

const enriched = setSession(context.active(), { sessionId: "sess-42" });

context.with(enriched, () => {
  const tracer = trace.getTracer("manual");
  const span = tracer.startSpan("manual-span");

  // Manually apply propagated attributes
  span.setAttributes(getAttributesFromContext(context.active()));

  span.end();
});
```

## Common Patterns

### Express/Koa Middleware

Set context attributes from request headers for all downstream spans:

```typescript
import { context } from "@opentelemetry/api";
import { setSession, setUser } from "@arizeai/openinference-core";

app.use((req, res, next) => {
  let ctx = context.active();

  const sessionId = req.headers["x-session-id"] as string;
  if (sessionId) {
    ctx = setSession(ctx, { sessionId });
  }

  const userId = req.headers["x-user-id"] as string;
  if (userId) {
    ctx = setUser(ctx, { userId });
  }

  context.with(ctx, () => next());
});
```

### Scoping Context to a Single Operation

```typescript
import { context } from "@opentelemetry/api";
import { setMetadata } from "@arizeai/openinference-core";

// Only this specific call gets the metadata
await context.with(
  setMetadata(context.active(), { experiment: "new-embeddings-v2" }),
  () => myTracedRetriever("search query"),
);

// This call does NOT have the metadata
await myTracedRetriever("another query");
```
