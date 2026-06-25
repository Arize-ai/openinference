# OpenInference Vercel

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-vercel.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-vercel)

This package provides utilities to ingest [Vercel AI SDK](https://github.com/vercel/ai) spans into platforms like [Arize](https://arize.com/) and [Phoenix](https://phoenix.arize.com/).

> Note: This package targets AI SDK v6 and is tested against v6 telemetry. Older versions (>= 3.3) are best-effort compatible.

## AI SDK Compatibility

| AI SDK version | Support level | Notes                                                                                                                                          |
| -------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| v6.x           | Targeted      | Emits `gen_ai.*` (OTel GenAI semconv) + `ai.*` (Vercel-specific). `@arizeai/openinference-vercel` prefers `gen_ai.*` and falls back to `ai.*`. |
| v5.x           | Best effort   | Telemetry primarily uses `ai.*`. Some standard `gen_ai.*`-derived mappings may be unavailable.                                                 |
| >= 3.3 and < 5 | Best effort   | Telemetry is experimental; attribute shapes may differ.                                                                                        |

## Installation

```shell
npm install --save @arizeai/openinference-vercel
```

You will also need to install OpenTelemetry and Vercel packages to your project.

```shell
npm i @opentelemetry/api @vercel/otel @opentelemetry/exporter-trace-otlp-proto @arizeai/openinference-semantic-conventions
```

## Usage

`@arizeai/openinference-vercel` provides a set of utilities to help you ingest Vercel AI SDK spans into platforms and works in conjunction with Vercel's OpenTelemetry support. To get started, you will need to add OpenTelemetry support to your Vercel project according to their [guide](https://vercel.com/docs/observability/otel-overview)

To process your Vercel AI SDK Spans add a `OpenInferenceSimpleSpanProcessor` or `OpenInferenceBatchSpanProcessor` to your OpenTelemetry configuration.

> [!NOTE]
> The `OpenInferenceSpanProcessor` does not handle the exporting of spans so you will pass it an [exporter](https://opentelemetry.io/docs/languages/js/exporters/) as a parameter.

```typescript
// instrumentation.ts
import { registerOTel } from "@vercel/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import {
  isOpenInferenceSpan,
  OpenInferenceSimpleSpanProcessor,
} from "@arizeai/openinference-vercel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

export function register() {
  registerOTel({
    serviceName: "phoenix-next-app",
    attributes: {
      // This is not required but it will ensure your traces get added to a specific project in Arize Phoenix
      [SEMRESATTRS_PROJECT_NAME]: "your-next-app",
    },
    spanProcessors: [
      new OpenInferenceSimpleSpanProcessor({
        exporter: new OTLPTraceExporter({
          headers: {
            // API key if you are sending it to Phoenix Cloud
            api_key: process.env["PHOENIX_API_KEY"] || "",
            // API key if you are sending it to local Phoenix
            Authorization: `Bearer ${process.env["PHOENIX_API_KEY"]}` || "",
          },
          url:
            process.env["PHOENIX_COLLECTOR_ENDPOINT"] || "https://app.phoenix.arize.com/v1/traces",
        }),
        spanFilter: (span) => {
          // Only export spans that are OpenInference to negate non-generative spans
          // This should be removed if you want to export all spans
          return isOpenInferenceSpan(span);
        },
      }),
    ],
  });
}
```

Now enable telemetry in your AI SDK calls by setting the `experimental_telemetry` parameter to `true`.

```typescript
const result = await generateText({
  model: openai("gpt-4-turbo"),
  prompt: "Write a short story about a cat.",
  experimental_telemetry: { isEnabled: true },
});
```

For details on Vercel AI SDK telemetry see the [Vercel AI SDK Telemetry documentation](https://sdk.vercel.ai/docs/ai-sdk-core/telemetry).

For more information on Vercel OpenTelemetry support see the [Vercel AI SDK Telemetry documentation](https://sdk.vercel.ai/docs/ai-sdk-core/telemetry).

## Reparenting orphaned spans

When you filter with `spanFilter: isOpenInferenceSpan`, only AI-related spans are
exported. But the highest-level AI span (e.g. `ai.generateText`, `ai.streamText`) is
frequently parented under a **non-AI span** — for example the HTTP/server span that
Next.js parents everything under. That parent is filtered out, leaving the AI span
**orphaned**: it references a parent that was never exported, so backends may not be
able to render the trace correctly.

Set `reparentOrphanedSpans: true` to detach (re-root) any AI span whose direct parent
is a non-AI span, so it becomes a trace root. Multiple sibling AI spans in the same
trace are each re-rooted, and AI spans nested under an AI parent keep their place.

```typescript
new OpenInferenceSimpleSpanProcessor({
  exporter,
  spanFilter: isOpenInferenceSpan,
  reparentOrphanedSpans: true, // default: false
});
```

The check is stateless — the parent span is read from the start-time context, so no
per-trace bookkeeping is kept.

If the re-rooted span is an `ai.*` framework wrapper that the package doesn't map to a
span kind (for example a per-turn span an agent framework emits on top of the AI SDK),
it would otherwise be kind-less and dropped by the filter. Such a root is tagged
`openinference.span.kind = AGENT` so it is kept as the trace root. This is matched by
shape (an unrecognized AI-like root), not by any specific framework span name.

`reparentOrphanedSpans` is opt-in (default `false`). It is intended for use alongside a
filter that drops non-AI parent spans — which is exactly the situation that orphans the
AI children in the first place: when the filter removes a non-AI parent, every AI span
beneath it is left pointing at a parent that was never exported. Reparenting re-roots
those children so the trace still renders. (Conversely, without such a filter the non-AI
parent is still exported, so there is nothing to orphan, and reparenting would only split
an otherwise-intact trace.)

## Propagating session and context attributes

The Vercel AI SDK creates its own spans, so — unlike the OpenInference instrumentors,
which build spans through an `OITracer` — this processor does **not** read the
OpenInference context. That means values you set with the
[`@arizeai/openinference-core`](https://www.npmjs.com/package/@arizeai/openinference-core)
helpers (`setSession`, `setUser`, `setMetadata`, `setTags`) never reach the exported AI
spans. A `session.id` set around a call like this would otherwise be lost:

```typescript
import { context } from "@opentelemetry/api";
import { setSession } from "@arizeai/openinference-core";

context.with(setSession(context.active(), { sessionId }), () =>
  streamText({ model, prompt, experimental_telemetry: { isEnabled: true } }),
);
```

Set `propagateContextAttributes: true` to copy every OpenInference attribute on the
start-time context (`session.id`, `user.id`, `metadata.*`, `tag.tags`, …) directly onto
each span at `onStart`, so traces group into sessions in Arize / Phoenix.

```typescript
new OpenInferenceSimpleSpanProcessor({
  exporter,
  spanFilter: isOpenInferenceSpan,
  reparentOrphanedSpans: true,
  propagateContextAttributes: true, // default: false
});
```

Because the values are written directly onto the span, they survive both
`reparentOrphanedSpans` re-rooting and export, and spans started in the same context
(child model/tool calls) inherit them. `propagateContextAttributes` is opt-in
(default `false`), so existing behavior is unchanged.
