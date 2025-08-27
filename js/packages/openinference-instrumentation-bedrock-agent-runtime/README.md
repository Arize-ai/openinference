# OpenInference Instrumentation for AWS Bedrock Agent Runtime

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-bedrock-agent-runtime.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-bedrock-agent-runtime)

This module provides automatic instrumentation for the [AWS Bedrock Agent Runtime Client](https://docs.aws.amazon.com/AWSJavaScriptSDK/v3/latest/client/bedrock-agent-runtime/) which may be used in conjunction with [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node).

## Installation

```shell
npm install --save @arizeai/openinference-instrumentation-bedrock-agent-runtime
```

## Usage

To load the Bedrock Agent Runtime instrumentation, specify it in the registerInstrumentations call along with any additional instrumentation you wish to enable.

```typescript
import { NodeTracerProvider } = from "@opentelemetry/sdk-trace-node";
import {
  BedrockAgentInstrumentation,
} = from "@arizeai/openinference-instrumentation-bedrock-agent-runtime";
import { registerInstrumentations } = "@opentelemetry/instrumentation";

const provider = new NodeTracerProvider();
provider.register();

registerInstrumentations({
  instrumentations: [new BedrockAgentInstrumentation()],
});
```

## Manual Instrumentation

For cases where import order is difficult to control, you can manually instrument the Bedrock Agent Runtime client:

```typescript
import { BedrockAgentInstrumentation } from "@arizeai/openinference-instrumentation-bedrock-agent-runtime";
import * as bedrockAgentRuntime from "@aws-sdk/client-bedrock-agent-runtime";

const instrumentation = new BedrockAgentInstrumentation();
instrumentation.manuallyInstrument(bedrockAgentRuntime);

// Now use the Bedrock Agent Runtime client as normal
const client = new bedrockAgentRuntime.BedrockAgentRuntimeClient({
  region: "us-east-1",
});
```

## Examples

To run an example, run the following commands:

```shell
cd js/packages/openinference-instrumentation-bedrock-agent-runtime
pnpm install
pnpm -r build
npx -y tsx examples/run-invoke-agent.ts
```

For more information on OpenTelemetry Node.js SDK, see the [OpenTelemetry Node.js SDK documentation](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/).

## Using a Custom Tracer Provider

You can specify a custom tracer provider when creating the Bedrock Agent Runtime instrumentation. This is useful when you want to use a non-global tracer provider or have more control over the tracing configuration.

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { BedrockAgentInstrumentation } from "@arizeai/openinference-instrumentation-bedrock-agent-runtime";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

// Create a custom tracer provider
const customTracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "my-bedrock-agent-project",
  }),
});

// Pass the custom tracer provider to the instrumentation
const bedrockAgentInstrumentation = new BedrockAgentInstrumentation({
  tracerProvider: customTracerProvider,
});

// Register the instrumentation with the custom tracer provider
registerInstrumentations({
  instrumentations: [bedrockAgentInstrumentation],
  tracerProvider: customTracerProvider,
});
```

Alternatively, you can set the tracer provider after creating the instrumentation:

```typescript
const bedrockAgentInstrumentation = new BedrockAgentInstrumentation();
bedrockAgentInstrumentation.setTracerProvider(customTracerProvider);
```

## Compatibility

`@arizeai/openinference-instrumentation-bedrock-agent-runtime` is compatible with the following versions of the `@aws-sdk/client-bedrock-agent-runtime` package:

| AWS SDK Version | OpenInference Instrumentation Version |
| --------------- | ------------------------------------- |
| ^3.0.0          | ^0.2.0                                |
