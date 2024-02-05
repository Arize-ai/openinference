# OpenInference JS

This is the JavaScript version of OpenInference, a framework for collecting traces from LLM applications.

> [!NOTE]
> Currently we only support OpenAI but we are working on adding support for other LLM frameworks and SDKs. If you are interested in contributing, please reach out to us by joining our slack community or opening an issue!

## Installation

OpenInference uses OpenTelemetry Protocol (OTLP) to send traces to a compatible backend (e.x. [arize-phoenix](<[https://git](https://github.com/Arize-ai/phoenix)>)). To use OpenInference, you will need to install the OpenTelemetry SDK and the OpenInference instrumentation for the LLM framework you are using.

Install the OpenTelemetry SDK:

```shell
npm install --save @opentelemetry/exporter-trace-otlp-http @opentelemetry/exporter-trace-otlp-proto @opentelemetry/resources @opentelemetry/sdk-trace-node
```

Install the OpenInference instrumentation you would like to use:

```shell
npm install --save @arizeai/openinference-instrumentation-openai
```

If you plan on manually instrumenting your application, you will also need to install the OpenInference Semantic Conventions:

```shell
npm install --save @arizeai/openinference-semantic-conventions
```

> [!NOTE]
> This example instruments OpenAI but you can replace `@arizeai/openinference-instrumentation-openai` with the instrumentation(s) of your choosing.

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

> [!WARNING]
> Note the above instrumentation must run before any other code in your application. This is because the instrumentation will only capture spans for the code that runs after the instrumentation is loaded. Typically this is done by requiring the instrumentation when running your application.
> `node -r ./path/to/instrumentation.js ./path/to/your/app.js`

## Examples

For more examples on how to use OpenInference, see the [examples](./examples) directory.

## Contributing

See [contributing guide](../CONTRIBUTING) for information on how to contribute to this project and the [JS Development Guide](./DEVELOPMENT.md) for setting up a development environment.
