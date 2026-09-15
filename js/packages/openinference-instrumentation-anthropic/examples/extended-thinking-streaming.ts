/* eslint-disable no-console */
import "./instrumentation";

import Anthropic from "@anthropic-ai/sdk";

async function main() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const stream = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    thinking: {
      type: "enabled",
      budget_tokens: 1024,
    },
    messages: [
      {
        role: "user",
        content: "What is 27 * 453? Think it through step by step.",
      },
    ],
    stream: true,
  });

  let thinkingText = "";
  let responseText = "";

  for await (const chunk of stream) {
    if (chunk.type === "content_block_delta") {
      if (chunk.delta.type === "thinking_delta") {
        thinkingText += chunk.delta.thinking;
      } else if (chunk.delta.type === "text_delta") {
        responseText += chunk.delta.text;
        process.stdout.write(chunk.delta.text);
      }
    }
  }

  console.log("\n\nThinking:", thinkingText);
  console.log("Response:", responseText);
}

main().catch(console.error);
