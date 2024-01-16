# OpenInference JS

This is the JavaScript version of OpenInference, a framework for collecting traces from LLM applications.

> [!NOTE]
> Currently we only support OpenAI but we are working on adding support for other LLM frameworks and SDKs. If you are interested in contributing, please reach out to us by joining our slack community or opening an issue!

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

## Contributing

See [contributing guide](../CONTRIBUTING) for information on how to contribute to this project and the [JS Development Guide](./DEVELOPMENT.md) for setting up a development environment.
