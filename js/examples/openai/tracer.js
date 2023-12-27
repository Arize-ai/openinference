"use strict";

const { registerInstrumentations } = require("@opentelemetry/instrumentation");
const {
    OpenAIInstrumentation,
} = require("@arizeai/openinference-instrumentation-openai");
const {
    ConsoleSpanExporter,
    SimpleSpanProcessor,
} = require("@opentelemetry/sdk-trace-base");
const { NodeTracerProvider } = require("@opentelemetry/sdk-trace-node");
const { Resource } = require("@opentelemetry/resources");
const {
    SemanticResourceAttributes,
} = require("@opentelemetry/semantic-conventions");

const provider = new NodeTracerProvider({
    resource: new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: "openai-service",
    }),
});

provider.addSpanProcessor(new SimpleSpanProcessor(new ConsoleSpanExporter()));
provider.register();

registerInstrumentations({
    instrumentations: [new OpenAIInstrumentation({})],
});
