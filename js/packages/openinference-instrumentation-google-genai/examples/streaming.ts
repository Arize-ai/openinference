/* eslint-disable no-console */
import "./instrumentation";

import { createInstrumentedGoogleGenAI } from "../src";

// Create an instrumented GoogleGenAI instance
const ai = createInstrumentedGoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY!,
});

async function main() {
  console.log("Streaming response example:\n");

  // Stream a response
  const stream = await ai.models.generateContentStream({
    model: "gemini-2.5-flash",
    contents: "Write a short story about a robot learning to paint",
  });

  // Process the stream
  for await (const chunk of stream) {
    if (chunk.candidates?.[0]?.content?.parts?.[0]?.text) {
      process.stdout.write(chunk.candidates[0].content.parts[0].text);
    }
  }

  console.log("\n\n--- Streaming complete ---");
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
