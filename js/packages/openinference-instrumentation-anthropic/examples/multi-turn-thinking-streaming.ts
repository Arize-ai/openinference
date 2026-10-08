/* eslint-disable no-console */
import "./instrumentation";

import Anthropic from "@anthropic-ai/sdk";
import type { MessageParam } from "@anthropic-ai/sdk/resources/messages";

async function main() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const messages: MessageParam[] = [
    {
      role: "user",
      content: "What is 27 * 453? Think it through step by step.",
    },
  ];

  // Turn 1 (non-streaming): get full response including thinking blocks + signatures
  // Thinking blocks must be passed back unmodified in multi-turn requests.
  const firstResponse = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    thinking: { type: "enabled", budget_tokens: 1024 },
    messages,
  });

  console.log("Turn 1:");
  for (const block of firstResponse.content) {
    if (block.type === "thinking") {
      console.log("  [thinking]", block.thinking.slice(0, 80), "...");
    } else if (block.type === "text") {
      console.log("  [text]", block.text);
    }
  }

  // Pass full response (including thinking blocks + signatures) back unmodified
  messages.push({ role: "assistant", content: firstResponse.content });
  messages.push({
    role: "user",
    content: "Now divide that result by 9 and explain your reasoning.",
  });

  // Turn 2 (streaming): stream the final answer
  console.log("\nTurn 2 (streaming):");
  const stream = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    thinking: { type: "enabled", budget_tokens: 1024 },
    messages,
    stream: true,
  });

  for await (const chunk of stream) {
    if (chunk.type === "content_block_delta" && chunk.delta.type === "text_delta") {
      process.stdout.write(chunk.delta.text);
    }
  }
  console.log();
}

main().catch(console.error);
