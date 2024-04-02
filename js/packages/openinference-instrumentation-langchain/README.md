# OpenInference Instrumentation for langchainjs

This module provides automatic instrumentation for [langchainjs](https://github.com/langchain-ai/langchainjs). which may be used in conjunction with [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node).

## Installation

```shell
npm install --save @arizeai/openinference-instrumentation-langchain
```

## Usage

To load the Langchain instrumentation, specify it in the registerInstrumentations call along with any additional instrumentation you wish to enable.

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { LangchainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";
import { registerInstrumentations } from "@opentelemetry/instrumentation";

const provider = new NodeTracerProvider();
provider.register();

registerInstrumentations({
  instrumentations: [new LangchainInstrumentation()],
});
```

For more information on OpenTelemetry Node.js SDK, see the [OpenTelemetry Node.js SDK documentation](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/).
