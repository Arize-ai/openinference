# OpenInference Instrumentation for OpenAI Node.js SDK

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-openai.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-openai)

This module provides automatic instrumentation for the [OpenAI Node.js SDK](https://github.com/openai/openai-node) which may be used in conjunction with [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node).

## Installation

```shell
npm install --save @arizeai/openinference-instrumentation-openai
```

## Usage

To load the OpenAI instrumentation, specify it in the registerInstrumentations call along with any additional instrumentation you wish to enable.

```typescript
const { NodeTracerProvider } = require("@opentelemetry/sdk-trace-node");
const {
  OpenAIInstrumentation,
} = require("@arizeai/openinference-instrumentation-openai");
const { registerInstrumentations } = require("@opentelemetry/instrumentation");

const provider = new NodeTracerProvider();
provider.register();

registerInstrumentations({
  instrumentations: [new OpenAIInstrumentation()],
});
```

## Examples

To run an example, run the following commands:

```shell
cd js/packages/openinference-instrumentation-openai
pnpm install
pnpm -r build
npx -y tsx examples/chat.ts # or responses.ts, embed.ts, etc
```

For more information on OpenTelemetry Node.js SDK, see the [OpenTelemetry Node.js SDK documentation](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/).

## Using a Custom Tracer Provider

You can specify a custom tracer provider when creating the OpenAI instrumentation. This is useful when you want to use a non-global tracer provider or have more control over the tracing configuration.

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { OpenAIInstrumentation } from "@arizeai/openinference-instrumentation-openai";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// Create a custom tracer provider
const customTracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "my-openai-project",
  }),
});

// Pass the custom tracer provider to the instrumentation
const openaiInstrumentation = new OpenAIInstrumentation({
  tracerProvider: customTracerProvider,
});

// Register the instrumentation with the custom tracer provider
registerInstrumentations({
  instrumentations: [openaiInstrumentation],
  tracerProvider: customTracerProvider,
});
```

Alternatively, you can set the tracer provider after creating the instrumentation:

```typescript
const openaiInstrumentation = new OpenAIInstrumentation();
openaiInstrumentation.setTracerProvider(customTracerProvider);
```

## Compatibility

`@arizeai/openinference-instrumentation-openai` is compatible with the following versions of the `openai` package:

| OpenAI Version | OpenInference Instrumentation Version |
| -------------- | ------------------------------------- |
| ^6.0.0         | ^4.0.0                                |
| ^5.0.0         | ^3.0.0                                |
| ^4.0.0         | ^2.0.0                                |
