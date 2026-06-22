# Per-Step Runtime Context

## Overview

Eve's `step.started` event fires before each model call within a turn. Returning
a `runtimeContext` object from the handler merges those key-value pairs onto the
model-call span as OTel attributes, making them visible in Phoenix.

This is useful for attaching request-scoped data — user IDs, ticket numbers,
channel metadata — that is not available at agent startup time.

## Basic Usage

```typescript
import { defineInstrumentation } from "eve/instrumentation";
import { registerOTel } from "@vercel/otel";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { isOpenInferenceSpan, OpenInferenceSimpleSpanProcessor } from "@arizeai/openinference-eve";

export default defineInstrumentation({
  setup: ({ agentName }) =>
    registerOTel({
      serviceName: agentName,
      spanProcessors: [
        new OpenInferenceSimpleSpanProcessor({
          exporter: new OTLPTraceExporter({
            url: process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces",
          }),
          spanFilter: isOpenInferenceSpan,
        }),
      ],
    }),

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

## `step.started` Input

The `input` parameter has this shape (from the Eve SDK):

```typescript
{
  turn: {
    id: string;
    sequence: number;   // 0-based turn counter within the session
  };
  step: {
    index: number;      // 0-based step counter within the turn
  };
  channel: ChannelUnion; // narrowed with isChannel()
}
```

## Channel-Specific Context

Use `isChannel` from `eve/instrumentation` to narrow the channel union and
access channel-specific metadata:

```typescript
import { defineInstrumentation, isChannel } from "eve/instrumentation";
// import supportChannel from "./channels/support.js";

export default defineInstrumentation({
  setup: ({ agentName }) => registerOTel({ /* ... */ }),

  events: {
    "step.started"(input) {
      // Only attach metadata when the request comes through the support channel
      // if (!isChannel(input.channel, supportChannel)) {
      //   return undefined;
      // }
      // return {
      //   runtimeContext: {
      //     "support.channel_id": input.channel.metadata.channelId ?? "",
      //     "support.user_id": input.channel.metadata.triggeringUserId ?? "",
      //   },
      // };

      // Generic fallback: tag every span with turn/step indices
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

## How `runtimeContext` Attributes Appear in Phoenix

Attributes from `runtimeContext` are set on the model-call span directly as
plain OTel attributes. They are preserved as-is (not under `metadata.*`) and
appear alongside the `eve.*` → `metadata.*` attributes in the span detail view.

For example, `{ "app.user_id": "u-42" }` results in:

```
app.user_id = "u-42"
```

on the `ai.streamText` span for that step.

## Returning `undefined`

Returning `undefined` (or nothing) from `step.started` is valid — no
`runtimeContext` is applied to that step.
