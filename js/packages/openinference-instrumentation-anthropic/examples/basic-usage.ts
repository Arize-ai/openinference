import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { registerInstrumentations } from "@opentelemetry/instrumentation";
import { AnthropicInstrumentation } from "@arizeai/openinference-instrumentation-anthropic";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import Anthropic from "@anthropic-ai/sdk";

// Configure the tracer provider
const provider = new NodeTracerProvider();
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new OTLPTraceExporter({
      url: "http://localhost:6006/v1/traces", // Phoenix endpoint
    }),
  ),
);
provider.register();

// Register the Anthropic instrumentation
registerInstrumentations({
  instrumentations: [new AnthropicInstrumentation()],
});

async function main() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  // Simple message
  const message = await anthropic.messages.create({
    model: "claude-3-sonnet-20240229",
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
