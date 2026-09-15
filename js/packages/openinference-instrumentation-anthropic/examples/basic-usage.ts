/* eslint-disable no-console */
import "./instrumentation";

import Anthropic from "@anthropic-ai/sdk";

async function main() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  // Simple message
  const message = await anthropic.messages.create({
    model: "claude-opus-5",
    max_tokens: 1000,
    messages: [
      {
        role: "user",
        content: "What is the capital of France?",
      },
    ],
  });

  console.log("Response:", message.content[0]);
}

main().catch(console.error);
