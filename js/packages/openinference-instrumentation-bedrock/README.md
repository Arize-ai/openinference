# OpenInference Instrumentation for AWS Bedrock JS SDK

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-bedrock.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-bedrock)

This module provides automatic instrumentation for [AWS Bedrock Runtime](https://docs.aws.amazon.com/AWSJavaScriptSDK/v3/latest/Package/-aws-sdk-client-bedrock/), which may be used in conjunction with [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node).

## Installation

```shell
npm install --save @arizeai/openinference-instrumentation-bedrock
```

## Usage

To load the Bedrock instrumentation, specify it in the registerInstrumentations call along with any additional instrumentation you wish to enable.

```typescript
const { NodeTracerProvider } = require("@opentelemetry/sdk-trace-node");
const {
  BedrockInstrumentation,
} = require("@arizeai/openinference-instrumentation-bedrock");
const { registerInstrumentations } = require("@opentelemetry/instrumentation");

const provider = new NodeTracerProvider();
provider.register();

registerInstrumentations({
  instrumentations: [new BedrockInstrumentation()],
});
```

## Examples

To run an example, run the following commands:

```shell
cd js/packages/openinference-instrumentation-bedrock
pnpm install
pnpm -r build
npx tsx scripts/validate-invoke-model.ts # or validate-converse-comprehensive.ts
```

For more information on OpenTelemetry Node.js SDK, see the [OpenTelemetry Node.js SDK documentation](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/).

## Using a Custom Tracer Provider

You can specify a custom tracer provider when creating the Bedrock instrumentation. This is useful when you want to use a non-global tracer provider or have more control over the tracing configuration.

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { BedrockInstrumentation } from "@arizeai/openinference-instrumentation-bedrock";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// Create a custom tracer provider
const customTracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "my-bedrock-project",
  }),
});

// Pass the custom tracer provider to the instrumentation
const bedrockInstrumentation = new BedrockInstrumentation({
  tracerProvider: customTracerProvider,
});

// Register the instrumentation with the custom tracer provider
registerInstrumentations({
  instrumentations: [bedrockInstrumentation],
  tracerProvider: customTracerProvider,
});
```

Alternatively, you can set the tracer provider after creating the instrumentation:

```typescript
const bedrockInstrumentation = new BedrockInstrumentation();
bedrockInstrumentation.setTracerProvider(customTracerProvider);
```

## Compatibility

`@arizeai/openinference-instrumentation-bedrock` is compatible with the following versions of the `@aws-sdk/client-bedrock-runtime` package:

| AWS SDK Version | OpenInference Instrumentation Version |
| --------------- | ------------------------------------- |
| ^3.0.0          | ^0.2.0                                |
