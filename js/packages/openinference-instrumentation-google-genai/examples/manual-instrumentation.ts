/**
 * There are some times such as in lambdas or in server actions inside of vercel where you cannot tap into the automatic instrumentation.
 * This file shows an example of how if GoogleGenAI is already imported, you can manually instrument it after it's been created.
 */

import { SEMRESATTRS_PROJECT_NAME } from "@arizeai/openinference-semantic-conventions";

import { diag, DiagConsoleLogger, DiagLogLevel } from "@opentelemetry/api";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { Resource } from "@opentelemetry/resources";
import {
  NodeTracerProvider,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-node";

import { GoogleGenAIInstrumentation } from "../src";

import { GoogleGenAI } from "@google/genai"; // Note that GoogleGenAI is imported before instrumentation

// For troubleshooting, set the log level to DiagLogLevel.DEBUG
diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.INFO);

const provider = new NodeTracerProvider({
  resource: new Resource({
    [SEMRESATTRS_PROJECT_NAME]: "google-genai-service",
  }),
  spanProcessors: [
    new SimpleSpanProcessor(
      new OTLPTraceExporter({
        url: "http://localhost:6006/v1/traces",
      }),
    ),
  ],
});

provider.register();

// eslint-disable-next-line no-console
console.log("Creating GoogleGenAI instance...");

// Create the GoogleGenAI instance first
const ai = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY!,
});

// Then manually instrument it
const instrumentation = new GoogleGenAIInstrumentation();
instrumentation.instrumentInstance(ai);

// eslint-disable-next-line no-console
console.log("GoogleGenAI instance instrumented");

// Use the AI instance normally - all calls are now traced
ai.models
  .generateContent({
    model: "gemini-2.5-flash",
    contents: "Write a haiku about programming",
  })
  .then((response) => {
    // eslint-disable-next-line no-console
    console.log(response.text);
  });
