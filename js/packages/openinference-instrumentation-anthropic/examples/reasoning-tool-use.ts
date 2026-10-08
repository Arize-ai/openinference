/* eslint-disable no-console */
import "./instrumentation";

import Anthropic from "@anthropic-ai/sdk";
import type { MessageParam } from "@anthropic-ai/sdk/resources/messages";

const tools: Anthropic.Tool[] = [
  {
    name: "get_weather",
    description: "Get the current weather for a location",
    input_schema: {
      type: "object",
      properties: {
        location: {
          type: "string",
          description: "The city and state, e.g. San Francisco, CA",
        },
      },
      required: ["location"],
    },
  },
];

async function main() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const messages: MessageParam[] = [
    {
      role: "user",
      content: "What is the weather in San Francisco?",
    },
  ];

  const firstResponse = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    thinking: { type: "enabled", budget_tokens: 1024 },
    tools,
    messages,
  });

  console.log("Turn 1 content blocks:");
  for (const block of firstResponse.content) {
    if (block.type === "thinking") {
      console.log("  [thinking]", block.thinking.slice(0, 80), "...");
    } else if (block.type === "tool_use") {
      console.log("  [tool_use]", block.name, JSON.stringify(block.input));
    } else if (block.type === "text") {
      console.log("  [text]", block.text);
    }
  }

  messages.push({ role: "assistant", content: firstResponse.content });

  const toolUseBlock = firstResponse.content.find((b) => b.type === "tool_use");
  if (!toolUseBlock || toolUseBlock.type !== "tool_use") {
    console.log("No tool call in response, stop_reason:", firstResponse.stop_reason);
    return;
  }

  messages.push({
    role: "user",
    content: [
      {
        type: "tool_result",
        tool_use_id: toolUseBlock.id,
        content: "Sunny, 68°F with light breeze",
      },
    ],
  });

  const secondResponse = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    thinking: { type: "enabled", budget_tokens: 1024 },
    tools,
    messages,
  });

  console.log("\nTurn 2 content blocks:");
  for (const block of secondResponse.content) {
    if (block.type === "thinking") {
      console.log("  [thinking]", block.thinking.slice(0, 80), "...");
    } else if (block.type === "text") {
      console.log("  [text]", block.text);
    }
  }
}

main().catch(console.error);
