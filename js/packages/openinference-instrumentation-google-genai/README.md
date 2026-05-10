# OpenInference Instrumentation for Google GenAI

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-google-genai.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-google-genai)

This package provides OpenTelemetry instrumentation for the [@google/genai](https://www.npmjs.com/package/@google/genai) SDK, enabling automatic tracing of Google Gemini API calls for observability and monitoring.

## Installation

```bash
npm install @arizeai/openinference-instrumentation-google-genai
```

## Usage

To enable instrumentation, register `GoogleGenAIInstrumentation` once at application startup. After it is registered, every `new GoogleGenAI({ ... })` constructed in your code is auto-instrumented — you do not need to change how you import or use the SDK.

We recommend isolating instrumentation setup into its own file (e.g. `instrumentation.ts`) and importing it as the very first thing at your application's entry point:

```typescript
// instrumentation.ts
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import { NodeTracerProvider, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-node";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { GoogleGenAIInstrumentation } from "@arizeai/openinference-instrumentation-google-genai";

const provider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "my-app",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces", // Phoenix or your OTLP endpoint
      }),
    ),
  ],
});
provider.register();

registerInstrumentations({
  instrumentations: [new GoogleGenAIInstrumentation()],
});
```

```typescript
// app.ts
import "./instrumentation";

import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });

const response = await ai.models.generateContent({
  model: "gemini-2.5-flash",
  contents: "What is the capital of France?",
});

console.log(response.text);
```

### Manual instrumentation (ESM / serverless)

In environments where module hooks are unavailable — for example pure ESM apps without `import-in-the-middle`, AWS Lambda, or some Vercel server actions — you can instrument an existing `GoogleGenAI` instance directly:

```typescript
import { GoogleGenAI } from "@google/genai";
import { GoogleGenAIInstrumentation } from "@arizeai/openinference-instrumentation-google-genai";

const instrumentation = new GoogleGenAIInstrumentation();
const ai = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });
instrumentation.instrumentInstance(ai);
```

## Configuration

### Trace configuration

You can mask sensitive information using `traceConfig`:

```typescript
import { GoogleGenAIInstrumentation } from "@arizeai/openinference-instrumentation-google-genai";

const instrumentation = new GoogleGenAIInstrumentation({
  traceConfig: {
    hideInputs: true,
    hideOutputs: true,
  },
});

registerInstrumentations({ instrumentations: [instrumentation] });
```

## Supported Operations

This instrumentation automatically traces the following Google Gen AI SDK operations:

### Models

- `models.generateContent()` — Text generation
- `models.generateContentStream()` — Streaming text generation
- `models.generateImages()` — Image generation

### Chats

- `chats.create()` — Create chat session
- `chat.sendMessage()` — Send message in chat
- `chat.sendMessageStream()` — Send message with streaming response

### Batches

- `batches.createEmbeddings()` — Batch embeddings job

## Captured Attributes

The instrumentation captures OpenInference semantic conventions including:

- **Span Kind**: `LLM`, `EMBEDDING`
- **Model Name**: e.g. `gemini-2.5-flash`
- **Input/Output**: Request and response content
- **Token Usage**: Prompt, completion, total, and cached token counts
- **Messages**: Multi-turn input messages with roles, function calls, and tool responses
- **Tools**: Function/tool definitions when using tool calling
- **System**: `vertexai`
- **Provider**: `google`

## Examples

See the [examples](./examples) directory for complete working examples:

- **[chat.ts](./examples/chat.ts)** — Basic text generation
- **[streaming.ts](./examples/streaming.ts)** — Streaming text generation
- **[chat-session.ts](./examples/chat-session.ts)** — Multi-turn chat conversations
- **[tools.ts](./examples/tools.ts)** — Function/tool calling
- **[embeddings.ts](./examples/embeddings.ts)** — Batch embeddings creation
- **[manual-instrumentation.ts](./examples/manual-instrumentation.ts)** — Manual instrumentation for special environments (lambdas, server actions)
- **[instrumentation.ts](./examples/instrumentation.ts)** — OpenTelemetry setup (shared by examples)

## Viewing Traces

### With Phoenix (Recommended)

1. Start Phoenix:

   ```bash
   docker run --pull=always -d --name arize-phoenix -p 6006:6006 arizephoenix/phoenix:latest
   ```

2. Open http://localhost:6006 in your browser.

3. Run your instrumented application — traces will appear in Phoenix.

### With other OTLP collectors

Configure the `OTLPTraceExporter` to point to your collector:

```typescript
new OTLPTraceExporter({ url: "http://your-collector:4318/v1/traces" });
```

## License

Apache-2.0

## Contributing

Contributions are welcome! Please see the main [OpenInference repository](https://github.com/Arize-ai/openinference) for contribution guidelines.
