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
npm install @arizeai/openinference-instrumentation-anthropic @anthropic-ai/sdk @opentelemetry/sdk-node @opentelemetry/exporter-otlp-http
```

Set up instrumentation in your application:

```typescript
import { NodeSDK } from '@opentelemetry/sdk-node';
import { AnthropicInstrumentation } from '@arizeai/openinference-instrumentation-anthropic';
import { OTLPTraceExporter } from '@opentelemetry/exporter-otlp-http';

// Configure the SDK with Anthropic instrumentation
const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter({
    url: 'http://localhost:6006/v1/traces', // Phoenix endpoint
  }),
  instrumentations: [
    new AnthropicInstrumentation({
      // Optional: configure trace settings
      traceConfig: {
        // hideInputs: true,
        // hideOutputs: true,
      }
    }),
  ],
});

// Initialize the SDK
sdk.start();

// Now use Anthropic as normal
import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const message = await anthropic.messages.create({
  model: 'claude-3-sonnet-20240229',
  max_tokens: 1000,
  messages: [{ role: 'user', content: 'Hello, Claude!' }],
});

console.log(message.content);
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

See the [examples](./examples) directory for more detailed usage examples.

## License

Apache-2.0
