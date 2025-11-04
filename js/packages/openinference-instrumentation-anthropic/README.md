# OpenInference Anthropic Instrumentation

JavaScript auto-instrumentation library for the [Anthropic](https://www.anthropic.com/) SDK

This package implements OpenInference tracing for the following Anthropic SDK methods:

- `messages.create` (including streaming)

These traces are fully OpenTelemetry compatible and can be sent to an OpenTelemetry collector for viewing, such as [Arize Phoenix](https://github.com/Arize-ai/phoenix).

## Installation

```shell
npm install @arizeai/openinference-instrumentation-anthropic
```

## Quickstart

Install required packages:

```shell
npm install @arizeai/openinference-instrumentation-anthropic @arizeai/openinference-semantic-conventions @anthropic-ai/sdk @opentelemetry/sdk-node @opentelemetry/exporter-trace-otlp-proto @opentelemetry/sdk-trace-node @opentelemetry/semantic-conventions
```

Set up instrumentation in your application:

```typescript
import { NodeSDK, resources } from "@opentelemetry/sdk-node";
import { AnthropicInstrumentation } from "@arizeai/openinference-instrumentation-anthropic";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import Anthropic from "@anthropic-ai/sdk";

// Configure the SDK with Anthropic instrumentation
const instrumentation = new AnthropicInstrumentation({
  // Optional: configure trace settings
  traceConfig: {
    // hideInputs: true,
    // hideOutputs: true,
  },
});
// necessary when instrumenting in an ESM environment
instrumentation.manuallyInstrument(Anthropic);
const sdk = new NodeSDK({
  spanProcessors: [
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces",
      }),
    ),
  ],
  resource: resources.resourceFromAttributes({
    [ATTR_SERVICE_NAME]: "anthropic-service",
    [SEMRESATTRS_PROJECT_NAME]: "anthropic-service",
  }),
  instrumentations: [instrumentation],
});

// Initialize the SDK
sdk.start();

// Now use Anthropic as normal

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const message = await anthropic.messages.create({
  model: "claude-3-5-haiku-latest",
  max_tokens: 1000,
  messages: [{ role: "user", content: "Hello, Claude!" }],
});

console.log(message.content);

sdk.shutdown();
```

## Configuration

The `AnthropicInstrumentation` constructor accepts the following options:

- `instrumentationConfig`: Standard OpenTelemetry instrumentation configuration
- `traceConfig`: OpenInference-specific trace configuration for masking/redacting sensitive information
- `tracerProvider`: Optional custom tracer provider

## Supported Features

- **Messages API**: Full support for `anthropic.messages.create()`
- **Streaming**: Automatic handling of streaming responses
- **Tool Use**: Captures tool/function calling information
- **Token Usage**: Records input/output token counts when available
- **Error Handling**: Proper error recording and span status management

## Semantic Conventions

This instrumentation follows the [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md) for LLM observability, capturing:

- Model name and provider information
- Input/output messages and content
- Token usage statistics
- Tool calling details
- Invocation parameters

## Examples

To run an example, run the following commands:

```shell
cd js/packages/openinference-instrumentation-anthropic
pnpm install
pnpm -r build
pnpx tsx examples/basic-usage.ts # or streaming.ts, tool-use.ts, etc
```

See the [examples](./examples) directory for more detailed usage examples.

## License

Apache-2.0
