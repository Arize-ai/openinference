# OpenInference TypeSafe AI Instrumentation

OpenTelemetry instrumentation for [`@typesafe-ai/sdk`](https://www.npmjs.com/package/@typesafe-ai/sdk).
Each `TypeSafeClient.systemOne` call produces one `LLM` span. `client.models.list()` is not instrumented.

Requires Node.js 20+ and `@typesafe-ai/sdk >=0.6.0 <0.7.0`.

## Installation

```sh
npm install @arizeai/openinference-instrumentation-typesafe @typesafe-ai/sdk
```

## Usage

Register the instrumentation **before** loading the SDK in CommonJS:

```ts
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { TypeSafeInstrumentation } from "@arizeai/openinference-instrumentation-typesafe";

registerInstrumentations({ instrumentations: [new TypeSafeInstrumentation()] });
```

For ESM, bundlers, or when the SDK is imported first, call `manuallyInstrument`:

```ts
import * as TypeSafe from "@typesafe-ai/sdk";
import { TypeSafeInstrumentation } from "@arizeai/openinference-instrumentation-typesafe";

const instrumentation = new TypeSafeInstrumentation();
instrumentation.manuallyInstrument(TypeSafe);

const client = new TypeSafe.TypeSafeClient();
const { data, requestId } = await client
  .systemOne({
    state: { document: "I was charged twice. Please fix this ASAP." },
    questions: {
      category: TypeSafe.choice("What is this ticket about?", {
        billing: null,
        technical: null,
        other: null,
      }),
    },
  })
  .withResponse();

console.log(data.answers.category, requestId);
```

The wrapped return value remains an SDK `APIPromise` (`await`, `withResponse()`, `map()`, etc.).

## Captured attributes

| Attribute | Value |
| --- | --- |
| Span name / kind | `TypeSafeClient.systemOne` / `LLM` |
| `llm.provider`, `llm.system` | `typesafe` |
| `llm.model_name` | Response model, else request or client default |
| `input.value` / `output.value` | Full JSON request and response (`application/json`) |
| `llm.token_count.*` | Prompt/completion when present; total when both exist |
| `llm.invocation_parameters` | Model and explicit timeout/retry overrides |
| `metadata.typesafe` | Request ID and question types/confidence |
| `http.response.status_code` | HTTP status on SDK API errors |

No `llm.input_messages` or `llm.output_messages` are emitted. State, questions, and answers live in the JSON payloads. Noul answers omit confidence.

Context metadata is preserved; `typesafe` is reserved for the instrumentor:

```json
{
  "workflow": "support-routing",
  "typesafe": {
    "request_id": "req_123",
    "questions": { "category": { "type": "choice", "confidence": 0.95 } }
  }
}
```

## Configuration

```ts
const instrumentation = new TypeSafeInstrumentation({
  tracerProvider, // optional; defaults to the global provider
  instrumentationConfig: { enabled: true },
  traceConfig: { hideInputs: true, hideOutputs: true },
});
```

Supports OpenInference context attributes, tracing suppression, and `TraceConfig` env vars.
`hideInputs` / `hideOutputs` redact payloads and related question/confidence metadata.
`hideInputMessages` / `hideOutputMessages` have no effect (no chat-message attributes).
Headers, credentials, and abort signals are never copied into span attributes.

## Examples

With Phoenix at `http://localhost:6006` and `TYPESAFE_API_KEY` set:

```sh
cd js
pnpm install --frozen-lockfile -r
pnpm --filter '@arizeai/openinference-instrumentation-typesafe...' run build
cd packages/openinference-instrumentation-typesafe
pnpm exec tsx examples/basic-usage.ts
```

| Example | Description |
| --- | --- |
| [basic-usage.ts](examples/basic-usage.ts) | Classification with `withResponse()` |
| [all-question-types.ts](examples/all-question-types.ts) | Choice, noul, score, and structured inputs |
| [guardrail-routing.ts](examples/guardrail-routing.ts) | TypeSafe gates an OpenAI call (also needs `OPENAI_API_KEY`) |
