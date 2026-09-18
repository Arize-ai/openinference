# OpenInference Instrumentation for TypeSafe AI SDK

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-typesafe.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-typesafe)

This module provides automatic instrumentation for the [TypeSafe AI Node.js SDK](https://www.npmjs.com/package/@typesafe-ai/sdk) (`@typesafe-ai/sdk`), which may be used in conjunction with [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node).

Each `TypeSafeClient.systemOne` call produces one OpenInference `LLM` span with JSON `input.value` / `output.value`, model, and token usage. `client.models.list()` is not instrumented. Traces follow the [OpenInference semantic conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md) and can be sent to any OpenTelemetry-compatible backend such as [Arize Phoenix](https://github.com/Arize-ai/phoenix) or [Arize AX](https://arize.com/products/ax?utm_source=docs&utm_medium=web&utm_content=openinference).

Requires Node.js 20+.

## Installation

```shell
npm install --save @arizeai/openinference-instrumentation-typesafe @typesafe-ai/sdk
```

## Usage

Register the instrumentation **before** loading the SDK in CommonJS:

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { TypeSafeInstrumentation } from "@arizeai/openinference-instrumentation-typesafe";

const provider = new NodeTracerProvider();
provider.register();

registerInstrumentations({
  instrumentations: [new TypeSafeInstrumentation()],
});
```

For ESM, bundlers, or when the SDK is imported first, call `manuallyInstrument`:

```typescript
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

The `TypeSafeInstrumentation` constructor accepts:

- `instrumentationConfig` — standard OpenTelemetry instrumentation configuration
- `traceConfig` — OpenInference masking/redaction options (`hideInputs`, `hideOutputs`, …)
- `tracerProvider` — optional custom tracer provider (defaults to the global provider)

```typescript
const instrumentation = new TypeSafeInstrumentation({
  tracerProvider,
  instrumentationConfig: { enabled: true },
  traceConfig: {
    hideInputs: true,
    hideOutputs: true,
  },
});
```

## Examples

To run an example against local Phoenix (`http://localhost:6006`):

```shell
cd js/packages/openinference-instrumentation-typesafe
pnpm install
pnpm -r build
export TYPESAFE_API_KEY=...
pnpm exec tsx examples/basic-usage.ts          # single classification
pnpm exec tsx examples/all-question-types.ts   # choice / noul / score
pnpm exec tsx examples/guardrail-routing.ts    # TypeSafe + OpenAI child spans
```

Shared OpenTelemetry setup lives in [`examples/instrumentation.ts`](./examples/instrumentation.ts).

For more information on OpenTelemetry Node.js SDK, see the [OpenTelemetry Node.js SDK documentation](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/).

## Using a Custom Tracer Provider

You can specify a custom tracer provider when creating the instrumentation. This is useful when you want to use a non-global tracer provider or have more control over the tracing configuration.

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { TypeSafeInstrumentation } from "@arizeai/openinference-instrumentation-typesafe";

const customTracerProvider = new NodeTracerProvider({
  resource: resourceFromAttributes({
    [SEMRESATTRS_PROJECT_NAME]: "my-typesafe-project",
  }),
});

const instrumentation = new TypeSafeInstrumentation({
  tracerProvider: customTracerProvider,
});

registerInstrumentations({
  instrumentations: [instrumentation],
  tracerProvider: customTracerProvider,
});
```

Alternatively, set the tracer provider after creating the instrumentation:

```typescript
const instrumentation = new TypeSafeInstrumentation();
instrumentation.setTracerProvider(customTracerProvider);
```

## Compatibility

`@arizeai/openinference-instrumentation-typesafe` is compatible with the following versions of the `@typesafe-ai/sdk` package:

| TypeSafe SDK Version | OpenInference Instrumentation Version |
| -------------------- | ------------------------------------- |
| >=0.6.0              | ^0.1.0                                |

## License

Apache-2.0
