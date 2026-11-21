/* eslint-disable no-console */
import "./instrumentation";

import Anthropic from "@anthropic-ai/sdk";

async function main() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  const message = await anthropic.messages.create({
    model: "claude-opus-5-5",
    max_tokens: 2048,
    thinking: { type: "adaptive", display: "summarized" },
    messages: [
      {
        role: "user",
        content: "What is 27 * 453? Think it through step by step.",
      },
    ],
  });

  console.log("Response:", JSON.stringify(message.content, null, 2));
}

main().catch(console.error);
