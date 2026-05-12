/* eslint-disable no-console */
import "./instrumentation";

import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY!,
});

async function main() {
  console.log("Creating embeddings...\n");

  const result = await ai.batches.createEmbeddings({
    model: "text-embedding-004",
    requests: [
      { content: "Hello, world!" },
      { content: "How are you?" },
      { content: "OpenInference is awesome!" },
    ],
  });

  console.log("Batch job created:", result.name);
  console.log("State:", result.state);
  console.log("\n--- Note: Batch jobs are processed asynchronously ---");
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
