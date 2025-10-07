import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { AnthropicInstrumentation } from "../src/instrumentation";
import * as Anthropic from "@anthropic-ai/sdk";

// Configure the tracer provider FIRST
const provider = new NodeTracerProvider();
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new OTLPTraceExporter({
      url: "http://localhost:6006/v1/traces", // Phoenix endpoint
    }),
  ),
);
provider.register();

// Create and manually instrument Anthropic to avoid module loading order issues
const anthropicInstrumentation = new AnthropicInstrumentation();
anthropicInstrumentation.setTracerProvider(provider);
anthropicInstrumentation.manuallyInstrument(Anthropic.default || Anthropic);

async function main() {
  const anthropic = new Anthropic.default({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  // Simple message
  const message = await anthropic.messages.create({
    model: "claude-3-5-sonnet-20241022", // Current model
    max_tokens: 1000,
    messages: [
      {
        role: "user",
        content: "What is the capital of France?",
      },
    ],
  });

  // eslint-disable-next-line no-console
  console.log("Response:", message.content[0]);
}

// eslint-disable-next-line no-console
main().catch(console.error);
