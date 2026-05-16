# OpenInference Vercel

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-vercel.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-vercel)

This package provides utilities to ingest [Vercel AI SDK](https://github.com/vercel/ai) spans into platforms like [Arize](https://arize.com/) and [Phoenix](https://phoenix.arize.com/).

> Note: This package targets AI SDK v7 telemetry. Use the latest v2 release for AI SDK v6.

## AI SDK Compatibility

| AI SDK version | Support level | Notes                                                                                                                                            |
| -------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| v7.x           | Targeted      | Uses `@ai-sdk/otel` `OpenTelemetry`, which emits `gen_ai.*` spans by default. Optional supplemental `ai.*` attributes fill non-GenAI data gaps. |
| v6.x and older | Unsupported   | Use `@arizeai/openinference-vercel` v2.x.                                                                                                       |

## Installation

```shell
npm install --save @arizeai/openinference-vercel
```

You will also need to install OpenTelemetry, `ai`, and `@ai-sdk/otel` packages to your project.

```shell
npm i ai @ai-sdk/otel @opentelemetry/api @vercel/otel @opentelemetry/exporter-trace-otlp-proto @arizeai/openinference-semantic-conventions
```

## Usage

`@arizeai/openinference-vercel` provides a set of utilities to help you ingest Vercel AI SDK spans into platforms and works in conjunction with Vercel's OpenTelemetry support. To get started, you will need to add OpenTelemetry support to your Vercel project according to their [guide](https://vercel.com/docs/observability/otel-overview)

To process your Vercel AI SDK Spans add a `OpenInferenceSimpleSpanProcessor` or `OpenInferenceBatchSpanProcessor` to your OpenTelemetry configuration.

> [!NOTE]
> The `OpenInferenceSpanProcessor` does not handle the exporting of spans so you will pass it an [exporter](https://opentelemetry.io/docs/languages/js/exporters/) as a parameter.

```typescript
// instrumentation.ts
import { registerOTel } from "@vercel/otel";
import { registerTelemetry } from "ai";
import { OpenTelemetry } from "@ai-sdk/otel";
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

Once `registerTelemetry(new OpenTelemetry())` is called, AI SDK v7 telemetry is enabled by default. Use `telemetry` only for metadata such as `functionId` or to opt out.

```typescript
const result = await generateText({
  model: openai("gpt-4-turbo"),
  prompt: "Write a short story about a cat.",
  telemetry: { functionId: "story-agent" },
});
```

For details on Vercel AI SDK telemetry see the [Vercel AI SDK Telemetry documentation](https://ai-sdk.dev/v7/docs/ai-sdk-core/telemetry).

For more information on Vercel OpenTelemetry support see the [Vercel AI SDK Telemetry documentation](https://sdk.vercel.ai/docs/ai-sdk-core/telemetry).
