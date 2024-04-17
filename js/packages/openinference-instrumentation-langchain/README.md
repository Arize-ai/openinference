# OpenInference Instrumentation for LangChain.js

This module provides automatic instrumentation for [LangChain.js](https://github.com/langchain-ai/langchainjs). which may be used in conjunction with [@opentelemetry/sdk-trace-node](https://github.com/open-telemetry/opentelemetry-js/tree/main/packages/opentelemetry-sdk-trace-node).

## Installation

```shell
npm install --save @arizeai/openinference-instrumentation-langchain
```

## Usage

To load the Langchain instrumentation, manually instrument the `@langchain/core/callbacks/manager` module. The callbacks manager must be manually instrumented due to the non-traditional module structure in `@langchain/core`. Additional instrumentations can be registered as usual using the `registerInstrumentations` function.

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { LangChainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";
import * as CallbackManagerModule from "@langchain/core/callbacks/manager";

const provider = new NodeTracerProvider();
provider.register();

const lcInstrumentation = new LangChainInstrumentation();
// LangChain must be manually instrumented as it doesn't have a traditional module structure
lcInstrumentation.manuallyInstrument(CallbackManagerModule);
```

For more information on OpenTelemetry Node.js SDK, see the [OpenTelemetry Node.js SDK documentation](https://opentelemetry.io/docs/instrumentation/js/getting-started/nodejs/).
