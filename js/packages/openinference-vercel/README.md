# OpenInference Vercel

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-vercel.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-vercel)

This package provides utilities to transform [Vercel AI SDK](https://github.com/vercel/ai) OpenTelemetry spans into OpenInference spans for platforms like [Arize AX](https://arize.com/) and [Phoenix](https://phoenix.arize.com/).

> Note: This package targets AI SDK v7 telemetry. Use `@arizeai/openinference-vercel` v2.x for AI SDK v6.

## AI SDK Compatibility

| AI SDK version | Support level | Notes                                                                                                                                           |
| -------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| v7.x           | Supported     | Uses `@ai-sdk/otel` `OpenTelemetry`, which emits `gen_ai.*` spans by default. Optional supplemental `ai.*` attributes fill non-GenAI data gaps. |
| v6.x and older | Unsupported   | Use `@arizeai/openinference-vercel` v2.x. AI SDK v6 used `experimental_telemetry` and emitted a different span shape.                           |

AI SDK v7 and this package require Node.js 22 or newer and are ESM-only. Configure your application accordingly before upgrading.

## Installation

```shell
npm install --save @arizeai/openinference-vercel
```

You will also need OpenTelemetry, `ai`, `@ai-sdk/otel`, and the AI SDK provider package you use. The examples below use `@ai-sdk/openai`.

```shell
npm i @arizeai/openinference-vercel ai@^7 @ai-sdk/otel@^1 @ai-sdk/openai@^4 @opentelemetry/api @opentelemetry/exporter-trace-otlp-proto @opentelemetry/resources @opentelemetry/sdk-trace-base @opentelemetry/sdk-trace-node @opentelemetry/semantic-conventions @arizeai/openinference-semantic-conventions
```

For Next.js applications deployed on Vercel, also install `@vercel/otel`:

```shell
npm i @vercel/otel
```

## Usage

`@arizeai/openinference-vercel` provides span processors that convert AI SDK v7 telemetry into OpenInference attributes before spans are exported. To get started, add OpenTelemetry to your application and register AI SDK telemetry once at application startup.

For Next.js apps deployed on Vercel, `registerOTel` from `@vercel/otel` still registers the OpenTelemetry provider, resource attributes, exporters, and span processors. AI SDK v7's `registerTelemetry(new OpenTelemetry(...))` does not replace that setup; it configures the AI SDK's telemetry integration.

To process Vercel AI SDK spans, add `OpenInferenceSimpleSpanProcessor` or `OpenInferenceBatchSpanProcessor` to your OpenTelemetry configuration.

> [!NOTE]
> The `OpenInferenceSpanProcessor` does not handle the exporting of spans by itself, pass it an [exporter](https://opentelemetry.io/docs/languages/js/exporters/) as a parameter.

### TypeScript Application

For a standalone TypeScript or Node.js application exporting to Phoenix, create an instrumentation module and import it before your AI SDK calls run.

```typescript
// instrumentation.ts
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { OpenInferenceBatchSpanProcessor } from "@arizeai/openinference-vercel";

import { OpenTelemetry } from "@ai-sdk/otel";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { registerTelemetry } from "ai";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG.
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO);

const phoenixUrl = process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces";
const phoenixApiKey = process.env["PHOENIX_API_KEY"];

export const tracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: process.env["PHOENIX_PROJECT_NAME"] ?? "my-typescript-app",
  }),
  spanProcessors: [
    new OpenInferenceBatchSpanProcessor({
      exporter: new OTLPTraceExporter({
        url: phoenixUrl,
        headers: phoenixApiKey
          ? {
              api_key: phoenixApiKey,
              Authorization: `Bearer ${phoenixApiKey}`,
            }
          : undefined,
      }),
    }),
  ],
});

tracerProvider.register();

registerTelemetry(
  new OpenTelemetry({
    // Optional, but recommended for fuller OpenInference coverage.
    usage: true,
    providerMetadata: true,
    embedding: true,
    reranking: true,
    runtimeContext: true,
    headers: true,
    toolChoice: true,
    schema: true,
  }),
);
```

Then import the instrumentation module before using the AI SDK.

```typescript
import "./instrumentation";

import { openai } from "@ai-sdk/openai";
import { generateText } from "ai";

const result = await generateText({
  model: openai("gpt-4o-mini"),
  prompt: "Write a short story about a cat.",
  // Telemetry is enabled by default once OpenTelemetry is registered.
  // Use telemetry for per-call metadata or to opt out with isEnabled: false.
  telemetry: { functionId: "story-agent" },
});
```

### Next.js Application

```typescript
// instrumentation.ts
import { registerOTel } from "@vercel/otel";
import { registerTelemetry } from "ai";
import { OpenTelemetry } from "@ai-sdk/otel";
import {
  isOpenInferenceSpan,
  OpenInferenceSimpleSpanProcessor,
} from "@arizeai/openinference-vercel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

export function register() {
  const phoenixApiKey = process.env["PHOENIX_API_KEY"];

  registerTelemetry(
    new OpenTelemetry({
      // Optional, but recommended for fuller OpenInference coverage.
      usage: true,
      providerMetadata: true,
      embedding: true,
      reranking: true,
      runtimeContext: true,
      headers: true,
      toolChoice: true,
      schema: true,
    }),
  );

  registerOTel({
    serviceName: "phoenix-next-app",
    attributes: {
      // This is not required but it will ensure your traces get added to a specific project in Arize Phoenix
      [SEMRESATTRS_PROJECT_NAME]: "your-next-app",
    },
    spanProcessors: [
      new OpenInferenceSimpleSpanProcessor({
        exporter: new OTLPTraceExporter({
          headers: phoenixApiKey
            ? {
                api_key: phoenixApiKey,
                Authorization: `Bearer ${phoenixApiKey}`,
              }
            : undefined,
          url:
            process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "https://app.phoenix.arize.com/v1/traces",
        }),
        spanFilter: (span) => {
          // Remove this filter if you want to export non-generative spans too.
          return isOpenInferenceSpan(span);
        },
      }),
    ],
  });
}
```

Once `registerTelemetry(new OpenTelemetry())` is called, AI SDK v7 telemetry is enabled by default. Use `telemetry` only for metadata such as `functionId` or to opt out.

```typescript
const result = await generateText({
  model: openai("gpt-4-turbo"),
  prompt: "Write a short story about a cat.",
  telemetry: { functionId: "story-agent" },
});
```

To disable telemetry for a single call, set `telemetry: { isEnabled: false }`.

For details on AI SDK v7 telemetry, see the [AI SDK telemetry documentation](https://ai-sdk.dev/docs/ai-sdk-core/telemetry).

For more information on Vercel OpenTelemetry support, see the [Vercel OpenTelemetry guide](https://vercel.com/docs/observability/otel-overview).

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

`propagateContextAttributes` copies every OpenInference attribute on the start-time
context (`session.id`, `user.id`, `metadata.*`, `tag.tags`, …) directly onto each span at
`onStart`, so traces group into sessions in Arize / Phoenix. It is enabled by default; set
it to `false` to opt out.

```typescript
new OpenInferenceSimpleSpanProcessor({
  exporter,
  spanFilter: isOpenInferenceSpan,
  reparentOrphanedSpans: true,
  propagateContextAttributes: true, // default: true
});
```

Because the values are written directly onto the span, they survive both
`reparentOrphanedSpans` re-rooting and export, and spans started in the same context
(child model/tool calls) inherit them.
