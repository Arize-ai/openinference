/* eslint-disable no-console */
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { AnthropicInstrumentation } from "@arizeai/openinference-instrumentation-anthropic";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import Anthropic from "@anthropic-ai/sdk";

// Configure the tracer provider
const provider = new NodeTracerProvider();
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new OTLPTraceExporter({
      url: "http://localhost:6006/v1/traces",
    }),
  ),
);
provider.register();

// Register the Anthropic instrumentation
const anthropicInstrumentation = new AnthropicInstrumentation();
anthropicInstrumentation.setTracerProvider(provider);
anthropicInstrumentation.manuallyInstrument(Anthropic);

async function streamingExample() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const tools = [
    {
      name: "get_weather",
      description: "Get the current weather in a given location",
      input_schema: {
        type: "object" as const,
        properties: {
          location: {
            type: "string" as const,
            description: "The city and state, e.g. San Francisco, CA",
          },
        },
        required: ["location"],
      },
    },
  ];

  // Streaming message
  const stream = await anthropic.messages.create({
    model: "claude-3-5-sonnet-latest",
    max_tokens: 1000,
    tools,
    messages: [
      {
        role: "user",
        content:
          "Tell me a short story about a robot using the current weather.",
      },
    ],
    stream: true,
  });

  console.log("Streaming response:");
  for await (const chunk of stream) {
    if (
      chunk.type === "content_block_delta" &&
      chunk.delta.type === "text_delta"
    ) {
      process.stdout.write(chunk.delta.text);
    }
  }
  console.log("\n\nStream complete!");
}

streamingExample().catch(console.error);
