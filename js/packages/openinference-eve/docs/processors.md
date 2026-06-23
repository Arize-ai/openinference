# Span Processors

## Overview

`@arizeai/openinference-eve` exports two span processors that extend their
counterparts from `@arizeai/openinference-vercel`:

- `OpenInferenceSimpleSpanProcessor` — synchronous export on each span end
- `OpenInferenceBatchSpanProcessor` — buffers spans and exports in batches

Both processors:
1. Call `addEveAttributesToSpan` to map `eve.*` → OpenInference attributes
2. Delegate to the parent Vercel processor's `onEnd` to apply AI SDK mappings
3. Accept a `spanFilter` option to conditionally skip spans

## `OpenInferenceSimpleSpanProcessor`

```typescript
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-eve";

new OpenInferenceSimpleSpanProcessor({
  exporter: new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" }),
  spanFilter: isOpenInferenceSpan,
})
```

**When to use:** Development, testing, or low-volume production. Synchronous
export means the span is sent to the exporter before `onEnd` returns. Simple
and predictable, but adds latency to each span end.

## `OpenInferenceBatchSpanProcessor`

```typescript
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { isOpenInferenceSpan, OpenInferenceBatchSpanProcessor } from "@arizeai/openinference-eve";

new OpenInferenceBatchSpanProcessor({
  exporter: new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" }),
  spanFilter: isOpenInferenceSpan,
})
```

**When to use:** Production. Spans are buffered and sent in batches, reducing
per-request overhead. Call `provider.forceFlush()` before process exit to
ensure buffered spans are sent.

## `spanFilter`

Both processors accept an optional `spanFilter` function:

```typescript
type SpanFilter = (span: ReadableSpan) => boolean;
```

If provided, a span is only exported when the filter returns `true`.

### `isOpenInferenceSpan`

The most common filter — passes only spans that carry at least one
OpenInference attribute. In an Eve app this includes:

- `ai.eve.turn` (has `openinference.span.kind = AGENT` after processing)
- `ai.streamText` (has `ai.operationId`)
- `ai.streamText.doStream` (has `gen_ai.*` or `ai.*` attributes)
- `ai.toolCall` (has `ai.toolCall.name`)

Infrastructure spans (HTTP, DNS, database) are excluded.

```typescript
import { isOpenInferenceSpan } from "@arizeai/openinference-eve";

new OpenInferenceSimpleSpanProcessor({
  exporter,
  spanFilter: isOpenInferenceSpan,
})
```

### Custom Filters

Combine filters to add further constraints:

```typescript
const filter = (span: ReadableSpan) =>
  isOpenInferenceSpan(span) && span.attributes["eve.environment"] === "production";

new OpenInferenceSimpleSpanProcessor({ exporter, spanFilter: filter })
```

## Processor Options

Both processors accept `{ exporter, spanFilter? }`.
`OpenInferenceBatchSpanProcessor` additionally accepts `config` with OTel
`BufferConfig` options:

```typescript
new OpenInferenceBatchSpanProcessor({
  exporter,
  spanFilter: isOpenInferenceSpan,
  config: {
    maxQueueSize: 2048,
    maxExportBatchSize: 512,
    scheduledDelayMillis: 5000,
    exportTimeoutMillis: 30000,
  },
})
```

## Using with `registerOTel` (Vercel / Next.js)

```typescript
import { registerOTel } from "@vercel/otel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { isOpenInferenceSpan, OpenInferenceBatchSpanProcessor } from "@arizeai/openinference-eve";

registerOTel({
  serviceName: "my-eve-agent",
  spanProcessors: [
    new OpenInferenceBatchSpanProcessor({
      exporter: new OTLPTraceExporter({ url: process.env["PHOENIX_COLLECTOR_ENDPOINT"] }),
      spanFilter: isOpenInferenceSpan,
    }),
  ],
});
```

## Using with `NodeTracerProvider` (standalone Node.js)

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { isOpenInferenceSpan, OpenInferenceBatchSpanProcessor } from "@arizeai/openinference-eve";

const provider = new NodeTracerProvider({
  resource: new Resource({ [SEMRESATTRS_PROJECT_NAME]: "my-eve-agent" }),
  spanProcessors: [
    new OpenInferenceBatchSpanProcessor({
      exporter: new OTLPTraceExporter({ url: "http://localhost:6006/v1/traces" }),
      spanFilter: isOpenInferenceSpan,
    }),
  ],
});

provider.register();
```
