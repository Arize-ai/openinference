/* eslint-disable no-console */
import "./instrumentation";
import Anthropic from "@anthropic-ai/sdk";

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

  // Initial non-streaming message
  const response = await anthropic.messages.create({
    model: "claude-3-5-sonnet-latest",
    max_tokens: 1000,
    tools,
    messages: [
      {
        role: "user",
        content:
          "Tell me a two sentence story about a robot using the current weather in San Francisco.",
      },
    ],
    stream: false,
  });

  const stream = await anthropic.messages.create({
    model: "claude-3-5-sonnet-latest",
    max_tokens: 1000,
    tools,
    messages: [
      {
        role: "user",
        content:
          "Tell me a two sentence story about a robot using the current weather in San Francisco.",
      },
      {
        role: response.role,
        content: response.content,
      },
      {
        role: "user",
        content: [
          {
            type: "tool_result",
            tool_use_id: response.content.find(
              (block) => block.type === "tool_use",
            )
              ? response.content.find((block) => block.type === "tool_use")!.id!
              : "",
            content: "The weather in San Francisco is sunny, 72°F",
          },
        ],
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
