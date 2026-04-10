# Trace Configuration and Data Masking

## OITracer

`OITracer` wraps a standard OpenTelemetry `Tracer` to add two capabilities:

1. **Context attribute propagation** -- automatically reads session, user,
   metadata, tags, and other context attributes and merges them into every span.
2. **Data masking** -- applies configurable rules to redact or remove sensitive
   attributes before they reach your span exporter.

### Constructor

```typescript
new OITracer({ tracer, traceConfig? })
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `tracer` | `Tracer` | An OpenTelemetry Tracer instance |
| `traceConfig` | `TraceConfigOptions` | Optional masking configuration (see below) |

### How withSpan Uses OITracer

When you call `withSpan(fn, options)`:

- If you pass `options.tracer` as a raw OTel `Tracer`, it is auto-wrapped via
  `wrapTracer()` into an `OITracer` (no masking, just context propagation).
- If you pass `options.tracer` as an `OITracer`, it is used directly (with its
  masking config).
- If you omit `options.tracer`, `getTracer()` creates an `OITracer` from the
  current global tracer provider when the wrapped function is invoked (no
  masking).

Agent decision rule:

- Omit `tracer` when you want wrappers to follow later global provider changes
- Pass an explicit `OITracer` when you need masking or want the wrapper pinned
  to a specific tracer configuration

To enable masking, you must explicitly create an `OITracer` with a `traceConfig`:

```typescript
import { trace } from "@opentelemetry/api";
import { OITracer, withSpan } from "@arizeai/openinference-core";

const maskedTracer = new OITracer({
  tracer: trace.getTracer("my-service"),
  traceConfig: {
    hideInputs: true,
    hideOutputText: true,
  },
});

const traced = withSpan(myFn, {
  tracer: maskedTracer,
  name: "sensitive-operation",
});
```

### OISpan

`OISpan` wraps an OpenTelemetry `Span` and intercepts `setAttribute` and
`setAttributes` calls to apply masking rules. All other `Span` methods (end,
setStatus, recordException, etc.) delegate directly to the underlying span.

You rarely interact with `OISpan` directly -- it is created internally by
`OITracer.startActiveSpan()` and `OITracer.startSpan()`.

## TraceConfigOptions

All options default to `false` (nothing hidden). Set individual flags to `true`
to redact or remove specific data.

| Option | Type | Default | Effect |
|--------|------|---------|--------|
| `hideInputs` | boolean | false | Redacts `input.value` to `"__REDACTED__"`, removes `input.mime_type`, removes all `llm.input_messages.*` |
| `hideOutputs` | boolean | false | Redacts `output.value` to `"__REDACTED__"`, removes `output.mime_type`, removes all `llm.output_messages.*` |
| `hideInputMessages` | boolean | false | Removes all `llm.input_messages.*` attributes |
| `hideOutputMessages` | boolean | false | Removes all `llm.output_messages.*` attributes |
| `hideInputImages` | boolean | false | Removes image content from input messages (`.message_content.image.*`) |
| `hideInputText` | boolean | false | Redacts text content in input messages to `"__REDACTED__"` |
| `hideOutputText` | boolean | false | Redacts text content in output messages to `"__REDACTED__"` |
| `hideEmbeddingVectors` | boolean | false | Removes all `embedding.embeddings.*.embedding.vector` attributes |
| `base64ImageMaxLength` | number | 32000 | Redacts base64-encoded images longer than this character count |
| `hidePrompts` | boolean | false | Redacts `llm.prompts` attribute |

### Redacted vs Removed

Some rules **redact** the value (replace with `"__REDACTED__"`), while others
**remove** the attribute entirely (it won't appear in the span):

| Behavior | Options |
|----------|---------|
| **Redacts** value to `"__REDACTED__"` | `hideInputs` (input.value), `hideOutputs` (output.value), `hideInputText`, `hideOutputText`, `hidePrompts`, `base64ImageMaxLength` |
| **Removes** attribute entirely | `hideInputs` (messages, mime_type), `hideOutputs` (messages, mime_type), `hideInputMessages`, `hideOutputMessages`, `hideInputImages`, `hideEmbeddingVectors` |

## Environment Variables

Every option can also be set via environment variable. Environment variables use
string `"true"`/`"false"` for booleans and numeric strings for numbers.

| Environment Variable | Maps To | Type |
|---------------------|---------|------|
| `OPENINFERENCE_HIDE_INPUTS` | `hideInputs` | boolean |
| `OPENINFERENCE_HIDE_OUTPUTS` | `hideOutputs` | boolean |
| `OPENINFERENCE_HIDE_INPUT_MESSAGES` | `hideInputMessages` | boolean |
| `OPENINFERENCE_HIDE_OUTPUT_MESSAGES` | `hideOutputMessages` | boolean |
| `OPENINFERENCE_HIDE_INPUT_IMAGES` | `hideInputImages` | boolean |
| `OPENINFERENCE_HIDE_INPUT_TEXT` | `hideInputText` | boolean |
| `OPENINFERENCE_HIDE_OUTPUT_TEXT` | `hideOutputText` | boolean |
| `OPENINFERENCE_HIDE_EMBEDDING_VECTORS` | `hideEmbeddingVectors` | boolean |
| `OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH` | `base64ImageMaxLength` | number |
| `OPENINFERENCE_HIDE_PROMPTS` | `hidePrompts` | boolean |

### Precedence

```
TraceConfigOptions value  >  environment variable  >  default
```

If you pass `hideInputs: false` in `TraceConfigOptions`, it overrides
`OPENINFERENCE_HIDE_INPUTS=true` in the environment.

### Example

```bash
OPENINFERENCE_HIDE_INPUTS=true OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH=8000 node app.js
```

## generateTraceConfig

Merges `TraceConfigOptions` + environment variables + defaults into a fully
resolved `TraceConfig` object (all fields required). Used internally by `OITracer`.
Rarely needed directly.

```typescript
import { generateTraceConfig } from "@arizeai/openinference-core";

const config = generateTraceConfig({ hideInputs: true });
// config.hideInputs === true
// config.hideOutputs === false (default)
// config.base64ImageMaxLength === 32000 (default)
// ... all other fields filled in
```

## wrapTracer / getTracer

```typescript
import { getTracer, wrapTracer } from "@arizeai/openinference-core";

// Wrap an existing OTel tracer (no-op if already an OITracer)
const oiTracer = wrapTracer(existingTracer);

// Create an OITracer from the global provider
const oiTracer = getTracer("my-service");  // name defaults to "openinference-core"
```

## Complete Example: Production Setup with Masking

```typescript
import {
  SEMRESATTRS_PROJECT_NAME,
} from "@arizeai/openinference-semantic-conventions";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { trace } from "@opentelemetry/api";
import { OITracer, withSpan } from "@arizeai/openinference-core";

// 1. Set up the OTel provider
const provider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "my-llm-app",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(
      new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" }),
    ),
  ],
});
provider.register();

// 2. Create an OITracer with masking
const tracer = new OITracer({
  tracer: trace.getTracer("my-llm-app"),
  traceConfig: {
    hideInputImages: true,        // Remove image data from input messages
    base64ImageMaxLength: 8000,   // Truncate large base64 images
    hideEmbeddingVectors: true,   // Don't export raw embedding vectors
  },
});

// 3. Use with withSpan
const chat = withSpan(
  async (prompt: string) => {
    const response = await llm.chat(prompt);
    return response.text;
  },
  {
    tracer,
    name: "chat",
    kind: "LLM",
  },
);

await chat("What is OpenInference?");
```
