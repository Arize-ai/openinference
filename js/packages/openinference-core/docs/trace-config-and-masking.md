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
  `wrapTracer()` into an `OITracer` (context propagation, and masking from
  the `OPENINFERENCE_*` environment variables only).
- If you pass `options.tracer` as an `OITracer`, it is used directly (with its
  masking config).
- If you omit `options.tracer`, `getTracer()` creates an `OITracer` from the
  current global tracer provider when the wrapped function is invoked (masking
  from the `OPENINFERENCE_*` environment variables only).

Agent decision rule:

- Omit `tracer` when you want wrappers to follow later global provider changes
- Pass an explicit `OITracer` when you need masking or want the wrapper pinned
  to a specific tracer configuration

To configure masking in code, create an `OITracer` with a `traceConfig`:

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
| `hideLLMTools` | boolean | false | Removes the tool definitions advertised to the LLM (`llm.tools.*`) |
| `hideInputs` | boolean | false | Redacts `input.value` to `"__REDACTED__"`, removes `input.mime_type`, removes all `llm.input_messages.*`, removes all `input.images.*`, removes all `llm.tools.*` |
| `hideOutputs` | boolean | false | Redacts `output.value` to `"__REDACTED__"`, removes `output.mime_type`, removes all `llm.output_messages.*`, removes all `output.images.*` |
| `hideInputMessages` | boolean | false | Removes all `llm.input_messages.*` attributes |
| `hideOutputMessages` | boolean | false | Removes all `llm.output_messages.*` attributes |
| `hideInputImages` | boolean | false | Removes image content from input messages (`.message_content.image.*`) and all `input.images.*` |
| `hideInputText` | boolean | false | Redacts text content in input messages to `"__REDACTED__"` |
| `hideOutputText` | boolean | false | Redacts text content in output messages to `"__REDACTED__"` |
| `hideEmbeddingVectors` | boolean | false | Removes all `embedding.embeddings.*.embedding.vector` attributes |
| `base64ImageMaxLength` | number | 32000 | Redacts base64-encoded image URLs longer than this character count, in input messages and under `input.images.*` / `output.images.*` |
| `hidePrompts` | boolean | false | Redacts `llm.prompts` attribute |
| `enableGenAISemconv` | boolean | false | Not a masking option. Also emits OpenTelemetry GenAI `gen_ai.*` attributes (experimental, see [GenAI Semantic Conventions Export](#genai-semantic-conventions-export)) |

### Redacted vs Removed

Some rules **redact** the value (replace with `"__REDACTED__"`), while others
**remove** the attribute entirely (it won't appear in the span):

| Behavior | Options |
|----------|---------|
| **Redacts** value to `"__REDACTED__"` | `hideInputs` (input.value), `hideOutputs` (output.value), `hideInputText`, `hideOutputText`, `hidePrompts`, `base64ImageMaxLength` |
| **Removes** attribute entirely | `hideInputs` (messages, mime_type, images, tools), `hideOutputs` (messages, mime_type, images), `hideInputMessages`, `hideOutputMessages`, `hideInputImages`, `hideEmbeddingVectors`, `hideLLMTools` |

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
| `OPENINFERENCE_HIDE_LLM_TOOLS` | `hideLLMTools` | boolean |
| `OPENINFERENCE_ENABLE_GENAI_SEMCONV` | `enableGenAISemconv` | boolean |

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

## GenAI Semantic Conventions Export

**Experimental.** Set `enableGenAISemconv: true` (or
`OPENINFERENCE_ENABLE_GENAI_SEMCONV=true`) to make every span created through an
`OITracer` also carry OpenTelemetry
[GenAI semantic convention](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
`gen_ai.*` attributes alongside its OpenInference attributes. Because the
conversion lives in `OITracer`, every OpenInference instrumentation built on it
picks the setting up without code changes.

```bash
OPENINFERENCE_ENABLE_GENAI_SEMCONV=true node app.js
```

```typescript
const tracer = new OITracer({
  tracer: trace.getTracer("my-service"),
  traceConfig: { enableGenAISemconv: true },
});
```

The `gen_ai.*` attributes are derived when the span ends, from the OpenInference
attributes set on it:

- `gen_ai.*` attributes you set yourself are never overwritten.
- Derivation runs after data masking, so hidden content stays hidden: a redacted
  message is redacted in `gen_ai.input.messages` too, and a removed one is absent.
- Only attributes set through the `OISpan` are converted. Attributes written to
  the underlying OpenTelemetry span directly are not seen.
- The derived attributes count against the span's attribute limits. A span that
  has already reached its attribute count limit (128 by default) when it ends
  gets no `gen_ai.*` attributes, which can happen on LLM spans with long
  conversations. Raise `spanLimits.attributeCountLimit` on the tracer provider
  (or `OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT`) for those workloads.

| OpenInference | GenAI |
|---------------|-------|
| `openinference.span.kind` | `gen_ai.operation.name` (`chat`, `text_completion`, `embeddings`, `execute_tool`, `retrieval`, `invoke_agent`) |
| `llm.provider`, `llm.system` | `gen_ai.provider.name` |
| `session.id` | `gen_ai.conversation.id` |
| `agent.name` | `gen_ai.agent.name` |
| `llm.request.model_name`, `llm.model_name`, `embedding.model_name` | `gen_ai.request.model` |
| `llm.invocation_parameters` | `gen_ai.request.*` (temperature, top_p, top_k, max_tokens, penalties, seed, stream, stop_sequences, choice.count), `gen_ai.output.type` |
| `llm.token_count.*` | `gen_ai.usage.*` |
| `llm.input_messages.*`, `llm.prompts` | `gen_ai.input.messages` (JSON) |
| `llm.output_messages.*`, `llm.choices` | `gen_ai.output.messages` (JSON), `gen_ai.response.finish_reasons` |
| `llm.response.model_name`, `output.value` (JSON, LLM spans) | `gen_ai.response.model` |
| `output.value` (JSON, LLM spans) | `gen_ai.response.id` |
| `llm.tools.*` | `gen_ai.tool.definitions` (JSON) |
| `tool.*`, `output.value` (TOOL spans) | `gen_ai.tool.name`, `gen_ai.tool.type`, `gen_ai.tool.description`, `gen_ai.tool.call.id`, `gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` |
| `retrieval.documents.*`, `input.value` (RETRIEVER spans) | `gen_ai.retrieval.documents` (JSON), `gen_ai.retrieval.query.text` |
| `embedding.embeddings.*` (EMBEDDING spans) | `gen_ai.embeddings.dimension.count` |

Provider vocabularies are normalized on the way:

- **Request parameters** are read by their snake_case names (OpenAI, Anthropic) and their camelCase names (Bedrock Converse, Gemini), e.g. `max_tokens`, `max_completion_tokens`, `maxTokens` and `maxOutputTokens` all become `gen_ai.request.max_tokens`.
- **Finish reasons** are mapped to the GenAI values: `tool_calls`, `function_call` and `tool_use` become `tool_call`; `end_turn` and `stop_sequence` become `stop`; `max_tokens` becomes `length`; `refusal`, `content_filtered`, `guardrail_intervened` and `safety` become `content_filter`. Other reasons pass through lower-cased.
- **Tool definitions** in the OpenAI (`function`), Anthropic (`input_schema`) and Bedrock Converse (`toolSpec`) shapes are converted to `{ type, name, description, parameters }`.

### getGenAIAttributes

The conversion itself is exported for span processors and other code that does
not create spans through an `OITracer`. It returns the derived attributes and
leaves merging to the caller:

```typescript
import { getGenAIAttributes } from "@arizeai/openinference-core";

const genAIAttributes = getGenAIAttributes({
  "openinference.span.kind": "LLM",
  "llm.model_name": "gpt-4o",
  "llm.provider": "openai",
});
// {
//   "gen_ai.operation.name": "chat",
//   "gen_ai.provider.name": "openai",
//   "gen_ai.request.model": "gpt-4o",
// }
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
    hideInputImages: true,        // Remove image data from input messages and input.images
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
