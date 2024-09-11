/**
 * There are some times such as in lambdas or in server actions inside of vercel where you cannot tap into the automatic instrumentation.
 * This file shows an example of how if openai is already imported, you can manually instrument it after it's been imported.
 */

import * as openai from "openai"; // Note that openai is imported before the instrumentation
import { isPatched, OpenAIInstrumentation } from "../src";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";
import { Resource } from "@opentelemetry/resources";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { SEMRESATTRS_SERVICE_NAME } from "@opentelemetry/semantic-conventions";
import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { assert } from "console";

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);

const provider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_SERVICE_NAME]: "openai-service",
  }),
});

provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new OTLPTraceExporter({
      url: "http://localhost:6006/v1/traces",
    }),
  ),
);

provider.register();

// Make sure that openai is not patched
assert(isPatched() === false);
// eslint-disable-next-line no-console
console.log("OpenAI is not patched");

const oaiInstrumentor = new OpenAIInstrumentation();

oaiInstrumentor.manuallyInstrument(openai);

// Make sure that openai is patched
assert(isPatched() === true);
// eslint-disable-next-line no-console
console.log("OpenAI is patched");

// Initialize OpenAI

const client = new openai.OpenAI();

client.chat.completions
  .create({
    model: "gpt-3.5-turbo",
    messages: [{ role: "system", content: "You are a helpful assistant." }],
    max_tokens: 150,
    temperature: 0.5,
  })
  .then((response) => {
    // eslint-disable-next-line no-console
    console.log(response.choices[0].message.content);
  });
