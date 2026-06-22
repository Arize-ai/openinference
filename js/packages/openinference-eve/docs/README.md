# @arizeai/openinference-eve Documentation

## What is This Package?

`@arizeai/openinference-eve` bridges the [Eve AI framework](https://eve.dev) and
[OpenInference](https://arize-ai.github.io/openinference/) observability. It
provides span processors that:

1. Map Eve's `eve.*` runtime attributes to OpenInference semantic conventions
2. Assign the correct `openinference.span.kind` to each span in the Eve
   span hierarchy
3. Inherit all Vercel AI SDK span processing from `@arizeai/openinference-vercel`
   (LLM token counts, messages, model name, tool call attributes)

The result is traces visible in [Phoenix](https://phoenix.arize.com) and
[Arize Cloud](https://arize.com) that show session context, per-turn structure,
and LLM usage details.

## Docs and Source Code in node_modules

Once installed, everything is available locally:

```
node_modules/@arizeai/openinference-eve/src/     # Full TypeScript source
node_modules/@arizeai/openinference-eve/docs/    # This documentation
```

Coding agents can read these files directly — no internet access required.

## Documentation Guide

| Document | When to Read It |
|---|---|
| [span-hierarchy.md](./span-hierarchy.md) | Understanding how Eve spans map to OpenInference span kinds |
| [attribute-mapping.md](./attribute-mapping.md) | How `eve.*` attributes map to OpenInference conventions |
| [processors.md](./processors.md) | Choosing between Simple and Batch processors, using `spanFilter` |
| [step-context.md](./step-context.md) | Enriching spans with per-step runtime context via the `step.started` event |

## Minimal Setup

```typescript
// instrumentation.ts — drop this in your Eve project root
import { defineInstrumentation } from "eve/instrumentation";
import { registerOTel } from "@vercel/otel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import {
  isOpenInferenceSpan,
  OpenInferenceSimpleSpanProcessor,
} from "@arizeai/openinference-eve";

export default defineInstrumentation({
  setup: ({ agentName }) =>
    registerOTel({
      serviceName: agentName,
      resourceAttributes: {
        [SEMRESATTRS_PROJECT_NAME]: process.env["PHOENIX_PROJECT_NAME"] ?? agentName,
      },
      spanProcessors: [
        new OpenInferenceSimpleSpanProcessor({
          exporter: new OTLPTraceExporter({
            url:
              process.env["PHOENIX_COLLECTOR_ENDPOINT"] ??
              "http://localhost:6006/v1/traces",
            headers:
              process.env["PHOENIX_API_KEY"] != null
                ? { Authorization: `Bearer ${process.env["PHOENIX_API_KEY"]}` }
                : undefined,
          }),
          spanFilter: isOpenInferenceSpan,
        }),
      ],
    }),
});
```

## Package Architecture

Eve builds on the Vercel AI SDK, so this package extends
`@arizeai/openinference-vercel` rather than reimplementing it:

```
@arizeai/openinference-eve
  └── extends @arizeai/openinference-vercel
        └── uses @arizeai/openinference-semantic-conventions
```

The Eve processors override `onEnd` to call `addEveAttributesToSpan` before
delegating to the Vercel processor's `onEnd`. The Vercel processor has a guard
in `getOISpanKindFromAttributes`: if `openinference.span.kind` is already set,
it returns the existing value. This means setting the span kind for
`ai.eve.turn` spans in `addEveAttributesToSpan` is respected downstream.

## Source Code Map

```
src/
  index.ts                         # Exports everything + re-exports from vercel
  constants.ts                     # EveFunctionNameToSpanKindMap, EVE_ATTRIBUTE_KEYS
  utils.ts                         # addEveAttributesToSpan (core mapping logic)
  OpenInferenceSpanProcessor.ts    # OpenInferenceSimpleSpanProcessor,
                                   #   OpenInferenceBatchSpanProcessor
examples/
  instrumentation.ts               # Phoenix OTLP setup
  instrumentation-arize.ts         # Arize Cloud setup
  instrumentation-with-context.ts  # Per-step runtime context
  verify-spans.ts                  # Runnable end-to-end verification (no LLM key needed)
  README.md                        # Examples guide
```

## All Exports at a Glance

**Span Processors**
- `OpenInferenceSimpleSpanProcessor` — synchronous, extends Vercel's Simple processor
- `OpenInferenceBatchSpanProcessor` — batched, extends Vercel's Batch processor

**Utilities (re-exported from `@arizeai/openinference-vercel`)**
- `isOpenInferenceSpan(span)` — returns `true` for spans with OpenInference attributes
- `SpanFilter` — type alias for `(span: ReadableSpan) => boolean`
