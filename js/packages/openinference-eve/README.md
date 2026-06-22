# OpenInference Eve

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-eve.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-eve)

`@arizeai/openinference-eve` provides OpenTelemetry span processing for the
[Eve AI framework](https://eve.dev), translating Eve's `eve.*` runtime
attributes and span hierarchy into [OpenInference](https://arize-ai.github.io/openinference/)
semantic conventions compatible with [Phoenix](https://phoenix.arize.com) and
[Arize Cloud](https://arize.com).

Eve builds on the [Vercel AI SDK](https://github.com/vercel/ai), so this
package extends `@arizeai/openinference-vercel` rather than reimplementing it.
All Vercel AI SDK span mapping (LLM token counts, messages, model name) is
inherited automatically.

## Installation

```bash
npm install @arizeai/openinference-eve @opentelemetry/exporter-trace-otlp-proto
```

## Quick Start

Drop an `instrumentation.ts` file into your Eve agent project root:

```typescript
import { defineInstrumentation } from "eve/instrumentation";
import { registerOTel } from "@vercel/otel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-eve";

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
              process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces",
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

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP endpoint |
| `PHOENIX_PROJECT_NAME` | agent name | Project name in Phoenix |
| `PHOENIX_API_KEY` | — | API key for Phoenix Cloud |

## Span Hierarchy

Eve creates one `ai.eve.turn` root span per conversational turn on top of the
standard Vercel AI SDK spans:

```
ai.eve.turn                    → AGENT  (session.id extracted from eve.session.id)
└── ai.streamText              → AGENT  (per step, via Vercel AI SDK)
    ├── ai.streamText.doStream → LLM    (token counts, messages, model name)
    └── ai.toolCall            → TOOL   (tool name, args, result)
```

## Attribute Mapping

Eve injects `eve.*` context attributes onto every span. This processor maps
them to OpenInference conventions automatically:

| Eve attribute | OpenInference attribute |
|---|---|
| `eve.session.id` | `session.id` |
| `eve.version` | `metadata.eve.version` |
| `eve.environment` | `metadata.eve.environment` |
| `eve.turn.id` | `metadata.eve.turn.id` |
| `eve.turn.sequence` | `metadata.eve.turn.sequence` |
| `eve.step.index` | `metadata.eve.step.index` |
| `eve.channel.kind` | `metadata.eve.channel.kind` |

`ai.eve.turn` spans also receive `openinference.span.kind = AGENT`.

## Processors

### `OpenInferenceSimpleSpanProcessor`

Processes spans synchronously as they end. Good for development and low-volume
production use. Mirrors `OpenInferenceSimpleSpanProcessor` from
`@arizeai/openinference-vercel`.

```typescript
import { OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-eve";
```

### `OpenInferenceBatchSpanProcessor`

Buffers and exports spans in batches. Recommended for production to reduce
export overhead.

```typescript
import { OpenInferenceBatchSpanProcessor } from "@arizeai/openinference-eve";
```

Both processors accept the same options as their Vercel counterparts plus an
optional `spanFilter` to limit which spans are exported.

## `isOpenInferenceSpan`

Re-exported from `@arizeai/openinference-vercel`. Returns `true` for spans
that carry OpenInference attributes (`ai.eve.turn`, `ai.streamText.doStream`,
`ai.toolCall`, etc.). Pass it as `spanFilter` to skip non-generative spans:

```typescript
import { isOpenInferenceSpan } from "@arizeai/openinference-eve";

new OpenInferenceSimpleSpanProcessor({
  exporter,
  spanFilter: isOpenInferenceSpan,
})
```

## Enriching Spans with Per-Step Context

Eve's `step.started` event fires before each model call and lets you attach
custom runtime attributes via `runtimeContext`. These become OTel span
attributes and are visible in Phoenix:

```typescript
export default defineInstrumentation({
  setup: ({ agentName }) => registerOTel({ /* ... */ }),

  events: {
    "step.started"(input) {
      return {
        runtimeContext: {
          "app.turn_sequence": input.turn.sequence,
          "app.step_index": input.step.index,
        },
      };
    },
  },
});
```

## Examples

See `examples/` for ready-to-copy instrumentation files:

| File | Description |
|---|---|
| `instrumentation.ts` | Send spans to Phoenix (OTLP) |
| `instrumentation-arize.ts` | Send spans to Arize Cloud |
| `instrumentation-with-context.ts` | Enrich spans with per-step runtime context |
| `verify-spans.ts` | Verify the integration locally without an LLM API key |

## Docs and Source Code in node_modules

Once installed, the full source and documentation are available locally:

```
node_modules/@arizeai/openinference-eve/src/     # Source code
node_modules/@arizeai/openinference-eve/docs/    # Detailed documentation
```

Coding agents can read these directly without internet access.

## Documentation

- [docs/](./docs/README.md) — detailed reference for agents and humans
- [OpenInference JS](https://arize-ai.github.io/openinference/js/)
- [Source code](https://github.com/Arize-ai/openinference/tree/main/js/packages/openinference-eve)
