# OpenInference Instrumentation for Google GenAI

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-google-genai.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-google-genai)

This package provides OpenTelemetry instrumentation for the [@google/genai](https://www.npmjs.com/package/@google/genai) SDK, enabling automatic tracing of Google GenAI API calls for observability and monitoring.

## Installation

```bash
npm install @arizeai/openinference-instrumentation-google-genai
```

or

```bash
yarn add @arizeai/openinference-instrumentation-google-genai
```

or

```bash
pnpm add @arizeai/openinference-instrumentation-google-genai
```

## Usage

### Basic Usage with Helper Function (Recommended)

Due to ESM module limitations, the recommended approach is to use the `createInstrumentedGoogleGenAI` helper function:

```typescript
import { NodeTracerProvider, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { createInstrumentedGoogleGenAI } from "@arizeai/openinference-instrumentation-google-genai";

// Set up OpenTelemetry
const provider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "my-app",
  }),
});

provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new OTLPTraceExporter({
      url: "http://localhost:6006/v1/traces", // Phoenix or your OTLP endpoint
    })
  )
);

provider.register();

// Create an instrumented GoogleGenAI instance
const ai = createInstrumentedGoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY,
});

// Use the AI instance normally - all calls are automatically traced!
const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: "What is the capital of France?",
});

console.log(response.text);
```

### Manual Instrumentation

If you need more control, you can manually instrument existing instances:

```typescript
import { GoogleGenAI } from "@google/genai";
import { GoogleGenAIInstrumentation } from "@arizeai/openinference-instrumentation-google-genai";

// Set up OpenTelemetry (same as above)
// ...

// Create instrumentation instance
const instrumentation = new GoogleGenAIInstrumentation();

// Create your GoogleGenAI instance
const ai = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY,
});

// Instrument it
instrumentation.instrumentInstance(ai);

// Now use ai normally - all calls are traced
const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: "Hello!",
});
```

## Configuration

### Trace Configuration

You can configure trace behavior to mask sensitive information:

```typescript
import { createInstrumentedGoogleGenAI, GoogleGenAIInstrumentation } from "@arizeai/openinference-instrumentation-google-genai";

// Create instrumentation with config
const instrumentation = new GoogleGenAIInstrumentation({
  traceConfig: {
    hideInputs: true,  // Mask input content
    hideOutputs: true, // Mask output content
  },
});

// Use with helper function
const ai = createInstrumentedGoogleGenAI(
  {
    apiKey: process.env.GOOGLE_API_KEY,
  },
  instrumentation
);
```

## Supported Operations

This instrumentation automatically traces the following Google GenAI SDK operations:

### Models
- `models.generateContent()` - Text generation
- `models.generateContentStream()` - Streaming text generation
- `models.generateImages()` - Image generation

### Chats
- `chats.create()` - Create chat session
- `chat.sendMessage()` - Send message in chat
- `chat.sendMessageStream()` - Send message with streaming response

## Captured Attributes

The instrumentation captures OpenInference semantic conventions including:

- **Span Kind**: `LLM` for model operations, `CHAIN` for chat operations
- **Model Name**: The model being used (e.g., `gemini-2.5-flash`)
- **Input/Output**: Request and response content
- **Token Usage**: Prompt, completion, and total token counts
- **Messages**: Input and output messages with roles and content
- **Tools**: Function/tool definitions when using tool calling
- **System**: `vertexai`
- **Provider**: `google`

## Examples

See the [examples](./examples) directory for complete working examples:

- **[chat.ts](./examples/chat.ts)** - Basic text generation with helper function
- **[streaming.ts](./examples/streaming.ts)** - Streaming text generation
- **[chat-session.ts](./examples/chat-session.ts)** - Multi-turn chat conversations
- **[tools.ts](./examples/tools.ts)** - Function/tool calling
- **[embeddings.ts](./examples/embeddings.ts)** - Batch embeddings creation
- **[manual-instrumentation.ts](./examples/manual-instrumentation.ts)** - Manual instrumentation for special environments (lambdas, server actions)
- **[instrumentation.ts](./examples/instrumentation.ts)** - OpenTelemetry setup (shared by examples)

## Viewing Traces

### With Phoenix (Recommended)

1. Install Phoenix:
```bash
pip install arize-phoenix
```

2. Start Phoenix:
```bash
python -m phoenix.server.main serve
```

3. Open http://localhost:6006 in your browser

4. Run your instrumented application - traces will appear in Phoenix!

### With Other OTLP Collectors

Configure the `OTLPTraceExporter` to point to your collector:

```typescript
new OTLPTraceExporter({
  url: "http://your-collector:4318/v1/traces",
})
```

## Troubleshooting

### ESM Module Limitations

The `@google/genai` SDK uses ESM with immutable exports and defines methods as arrow function class properties rather than prototype methods. This prevents traditional OpenTelemetry auto-instrumentation from working.

**Solution**: Use the `createInstrumentedGoogleGenAI()` helper function or manually call `instrumentation.instrumentInstance()` after creating your `GoogleGenAI` instance.

### No Traces Appearing

1. Verify OpenTelemetry is set up correctly and provider is registered
2. Check that you're using `createInstrumentedGoogleGenAI()` or have called `instrumentInstance()`
3. Ensure your OTLP exporter URL is correct
4. Check console for any error messages

## License

Apache-2.0

## Contributing

Contributions are welcome! Please see the main [OpenInference repository](https://github.com/Arize-ai/openinference) for contribution guidelines.
