# OpenInference Instrumentation for OpenAI Node.js SDK

[![npm version](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-openai.svg)](https://badge.fury.io/js/@arizeai%2Fopeninference-instrumentation-openai)

This module provides automatic instrumentation for the [OpenAI Node.js SDK](https://github.com/openai/openai-node). which may be used in conjunction with [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node).

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

For more information on OpenTelemetry Node.js SDK, see the [OpenTelemetry Node.js SDK documentation](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/).
