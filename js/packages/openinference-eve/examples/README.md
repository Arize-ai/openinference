# Examples

## Eve project instrumentation files

These files are meant to be placed in the root of an Eve agent project. They
require the `eve` package to be installed in that project.

| File | Description |
|---|---|
| `instrumentation.ts` | Send spans to Phoenix (OTLP) |
| `instrumentation-arize.ts` | Send spans to Arize Cloud |
| `instrumentation-with-context.ts` | Enrich spans with per-step runtime context via the `step.started` event |

### How it wires up

Eve calls `setup()` once at startup. Inside, you register an OTel provider
using `registerOTel` from `@vercel/otel` and pass an
`OpenInferenceSimpleSpanProcessor` (or `OpenInferenceBatchSpanProcessor`) as
one of the `spanProcessors`.

The processor handles the full Eve span hierarchy automatically:

```
ai.eve.turn               → AGENT  (eve.session.id → session.id)
└── ai.streamText         → AGENT  (per step)
    └── ai.streamText.doStream → LLM   (token counts, messages, model name)
        └── ai.toolCall   → TOOL   (tool name, args, result)
```

Eve injects `eve.*` context attributes onto every span in the trace. The
processor extracts `eve.session.id` into the OpenInference `session.id`
attribute and maps the remaining `eve.*` keys under `metadata.*`.

### Quickstart

```bash
# Inside your Eve agent project
pnpm add @arizeai/openinference-eve @opentelemetry/exporter-trace-otlp-proto
```

Copy one of the instrumentation files above to your project root and
configure the environment variables shown at the top of the file.

## Verification script (no Eve SDK required)

`verify-spans.ts` manually recreates the span hierarchy Eve produces for a
two-step weather-agent turn and exports it to a Phoenix instance via OTLP.
Use it to confirm attribute mapping without needing an LLM API key or the
Eve runtime.

### Prereqs

- `OPENAI_API_KEY` is **not** required (no real LLM calls are made).
- A running Phoenix instance (default: `http://localhost:6006`).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `PHOENIX_COLLECTOR_ENDPOINT` | `http://localhost:6006/v1/traces` | OTLP endpoint |
| `PHOENIX_PROJECT_NAME` | `openinference-eve-verify` | Project name in Phoenix |
| `PHOENIX_API_KEY` | — | API key (optional, for Phoenix Cloud) |

### Run

From `packages/openinference-eve`:

```bash
# Against a local Phoenix instance
pnpx tsx examples/verify-spans.ts

# Against Phoenix Cloud
PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com/v1/traces \
PHOENIX_API_KEY=your-api-key \
pnpx tsx examples/verify-spans.ts
```

Open Phoenix and look for the `openinference-eve-verify` project. You should
see one trace with four spans:

| Span | `openinference.span.kind` | `session.id` |
|---|---|---|
| `ai.eve.turn` | `AGENT` | `sess_demo_001` |
| `ai.streamText` | `AGENT` | — |
| `ai.streamText.doStream` | `LLM` | — |
| `ai.toolCall` | `TOOL` | — |
