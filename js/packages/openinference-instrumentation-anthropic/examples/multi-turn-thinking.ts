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

  const firstResponse = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    thinking: {
      type: "enabled",
      budget_tokens: 1024,
    },
    messages,
  });
  console.log("First response:", JSON.stringify(firstResponse.content, null, 2));

  // Thinking blocks (including their signatures) must be passed back unmodified
  // in the conversation history for multi-turn requests with extended thinking.
  messages.push({ role: "assistant", content: firstResponse.content });
  messages.push({
    role: "user",
    content: "Now divide that result by 9 and explain your reasoning.",
  });

  const secondResponse = await anthropic.messages.create({
    model: "claude-sonnet-4-6",
    max_tokens: 2048,
    thinking: {
      type: "enabled",
      budget_tokens: 1024,
    },
    messages,
  });
  console.log("Second response:", JSON.stringify(secondResponse.content, null, 2));
}

main().catch(console.error);
