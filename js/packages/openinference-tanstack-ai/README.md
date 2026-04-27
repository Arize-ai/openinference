# OpenInference TanStack AI

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-tanstack-ai.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-tanstack-ai)

This package provides an OpenInference middleware for [TanStack AI](https://tanstack.com/ai/latest/docs/getting-started/overview). It emits OpenTelemetry spans shaped according to the OpenInference specification so TanStack AI runs can be visualized in systems like [Arize](https://arize.com/) and [Phoenix](https://phoenix.arize.com/).

## Installation

```shell
npm install --save @arizeai/openinference-tanstack-ai @tanstack/ai
```

You will also need an OpenTelemetry setup in your application. For example:

```shell
npm install --save @arizeai/phoenix-otel
```

or:

```shell
npm install --save @opentelemetry/api @opentelemetry/sdk-trace-node @opentelemetry/exporter-trace-otlp-proto
```

Install the provider adapter you plan to use with TanStack AI as well, for example:

```shell
npm install --save @tanstack/ai-openai
```

## Usage

`@arizeai/openinference-tanstack-ai` exports `openInferenceMiddleware`, which plugs directly into TanStack AI's `middleware` option.

```ts
import { chat } from "@tanstack/ai";
import { openaiText } from "@tanstack/ai-openai";

import { openInferenceMiddleware } from "@arizeai/openinference-tanstack-ai";

const stream = chat({
  adapter: openaiText("gpt-4o-mini"),
  messages: [{ role: "user", content: "What is OpenInference?" }],
  middleware: [openInferenceMiddleware()],
});
```

The middleware works for both streaming and non-streaming TanStack AI calls.

```ts
const text = await chat({
  adapter: openaiText("gpt-4o-mini"),
  stream: false,
  systemPrompts: ["You are a concise technical explainer."],
  messages: [{ role: "user", content: "Explain OpenInference in one sentence." }],
  middleware: [openInferenceMiddleware()],
});
```

## Tracer Setup

This package uses your application's existing OpenTelemetry tracer provider and exporters. It does not export spans by itself.

> [!NOTE]
> Your instrumentation code should run before the middleware is applied. This ensures that the tracer provider is properly configured before the middleware starts emitting spans.

The recommended quick start is to pair it with `@arizeai/phoenix-otel`.

```ts
import { register } from "@arizeai/phoenix-otel";

register({
  projectName: "my-tanstack-ai-app",
  endpoint: process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces",
  apiKey: process.env["PHOENIX_API_KEY"],
});
```

If you already have a standard OpenTelemetry setup, that works as well. For example, with a local Phoenix collector, a minimal manual setup looks like this:

```ts
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

const tracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "my-tanstack-ai-app",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: process.env["PHOENIX_COLLECTOR_ENDPOINT"] ?? "http://localhost:6006/v1/traces",
        headers:
          process.env["PHOENIX_API_KEY"] == null
            ? undefined
            : {
                Authorization: `Bearer ${process.env["PHOENIX_API_KEY"]}`,
              },
      }),
    ),
  ],
});

tracerProvider.register();
```

## Custom Tracer

By default, the middleware uses the global tracer for this package. If your application already has a request-scoped or custom tracer, pass it explicitly.

```ts
import { trace } from "@opentelemetry/api";

const tracer = trace.getTracer("tanstack-ai-request");

const middleware = openInferenceMiddleware({ tracer });
```

This is useful when you want the middleware to participate in a specific tracer setup without relying on the global default.

## What Gets Traced

The middleware emits the following span structure for a TanStack AI run:

- One `AGENT` span for the overall `chat()` invocation
- One `LLM` span for each model turn
- One `TOOL` span for each executed tool call

For a tool loop, the trace will typically look like:

- `AGENT`
- `LLM 1`
- `TOOL`
- `LLM 2`

The `AGENT` span captures the top-level request and final response. The `LLM` spans capture provider/model metadata, input messages, output messages, tool definitions, and token counts. The `TOOL` spans capture tool names, arguments, outputs, and errors.

## Examples

This package includes example files in `examples/`:

- `examples/chat-with-tools.ts` - OpenAI example with one tool call
- `examples/anthropic-multi-tool.ts` - Anthropic example with multiple tool calls
- `examples/non-streaming-chat.ts` - Anthropic non-streaming example with a system prompt

See `examples/README.md` for setup and run commands.

## Notes

- This package is ESM-only because TanStack AI is ESM-only.
- The middleware works in both server and client environments, but client/server trace stitching depends on your application's context propagation setup.
