# OpenInference TypeSafe AI Instrumentation

OpenTelemetry instrumentation for [`@typesafe-ai/sdk`](https://www.npmjs.com/package/@typesafe-ai/sdk).
Each `TypeSafeClient.systemOne` call produces one `LLM` span. `client.models.list()` is not instrumented.

Requires Node.js 20+ and `@typesafe-ai/sdk >=0.6.0`.

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

## Configuration

```ts
const instrumentation = new TypeSafeInstrumentation({
  tracerProvider, // optional; defaults to the global provider
  instrumentationConfig: { enabled: true },
  traceConfig: { hideInputs: true, hideOutputs: true },
});
```
