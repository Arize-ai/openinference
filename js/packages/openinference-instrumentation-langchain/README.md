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

## Using a Custom Tracer Provider

You can specify a custom tracer provider when creating the LangChain instrumentation. This is useful when you want to use a non-global tracer provider or have more control over the tracing configuration.

```typescript
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";
import { LangChainInstrumentation } from "@arizeai/openinference-instrumentation-langchain";
import * as CallbackManagerModule from "@langchain/core/callbacks/manager";

// Create a custom tracer provider
const customTracerProvider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "my-langchain-project",
  }),
});

// Pass the custom tracer provider to the instrumentation
const lcInstrumentation = new LangChainInstrumentation({
  tracerProvider: customTracerProvider,
});

// Manually instrument the LangChain module
lcInstrumentation.manuallyInstrument(CallbackManagerModule);
```

Alternatively, you can set the tracer provider after creating the instrumentation:

```typescript
const lcInstrumentation = new LangChainInstrumentation();
lcInstrumentation.setTracerProvider(customTracerProvider);
```

## Compatibility

| @langchain/core Version | @arizeai/openinference-instrumentation-langchain Version |
| ----------------------- | -------------------------------------------------------- |
| ^1.0.0                  | ^4.0.0                                                   |
| ^0.3.0                  | ^4.0.0                                                   |

This package is only tested against the 1.X versions of `@langchain/core`. Older versions may work but are not officially supported. For full compatibility for the 0.X versions of LangChain.js, a dedicated package called `openinference-instrumentation-langchain-v0` is available.

## Deprecations

LangChain v0.1 was deprecated on 2025-03-02 due to security vulnerabilities in the core package.
