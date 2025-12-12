/* eslint-disable no-console */
import "./instrumentation";

import { createInstrumentedGoogleGenAI } from "../src";

// Create an instrumented GoogleGenAI instance
const ai = createInstrumentedGoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY!,
});

async function main() {
  console.log("Multi-turn chat session example:\n");

  // Create a chat session
  const chat = ai.chats.create({
    model: "gemini-2.5-flash",
    config: {
      systemInstruction: {
        parts: [
          { text: "You are a helpful assistant that speaks like a pirate." },
        ],
      },
    },
  });

  // Send first message
  console.log("User: Hello! What's your name?");
  const response1 = await chat.sendMessage({
    message: "Hello! What's your name?",
  });
  console.log("Assistant:", response1.text);
  console.log();

  // Send second message (history is maintained)
  console.log("User: Tell me a joke");
  const response2 = await chat.sendMessage({
    message: "Tell me a joke",
  });
  console.log("Assistant:", response2.text);
  console.log();

  // Send third message with streaming
  console.log("User: What's 2+2?");
  console.log("Assistant: ");
  const stream = await chat.sendMessageStream({
    message: "What's 2+2?",
  });

  for await (const chunk of stream) {
    if (chunk.candidates?.[0]?.content?.parts?.[0]?.text) {
      process.stdout.write(chunk.candidates[0].content.parts[0].text);
    }
  }

  console.log("\n\n--- Chat session complete ---");
  console.log("Note: All messages in this session were traced with context!");
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
