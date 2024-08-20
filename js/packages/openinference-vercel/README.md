# OpenInference Vercel AI SDK Span Processor

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-vercel)

This package provides a SpanProcessor which appends OpenInference attributes to AI spans created by [`@vercel/otel`](https://www.npmjs.com/package/@vercel/otel).

## Installation

```shell
npm install --save @arizeai/openinference-vercel
```

## Usage

To process your Vercel AI SDK Spans add the `VercelSpanProcessor` to your span processors along with any other span processors you wish to use. 

> Note: The `VercelSpanProcessor` does not handle the exporting of spans so you will want to pair it with another [span processor](https://opentelemetry.io/docs/languages/js/instrumentation/#picking-the-right-span-processor) that accepts an [exporter](https://opentelemetry.io/docs/languages/js/exporters/) as a parameter. 


```typescript
import { registerOTel } from "@vercel/otel";
import { VercelSpanProcessor } from "@arizeai/openinference-vercel";
import { OTLPHttpProtoTraceExporter } from "@vercel/otel";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";

export function register() {
  registerOTel({
    serviceName: "next-app",
    spanProcessors: [
      new VercelSpanProcessor(),
      new SimpleSpanProcessor(
        new OTLPHttpProtoTraceExporter({
          url: "http://localhost:6006/v1/traces",
        }),
      ),
    ],
  });
}
```

## Examples
To see an example go to the [Next.js OpenAI Telemetry Example](https://github.com/Arize-ai/openinference/tree/main/js/examples/next-openai-telemetry-app) in the examples directory of this repo.

For more information on Vercel OpenTelemetry support see the [Vercel AI SDK Telemetry documentation](https://sdk.vercel.ai/docs/ai-sdk-core/telemetry).

