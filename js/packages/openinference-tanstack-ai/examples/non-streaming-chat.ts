/* eslint-disable no-console */

import "./instrumentation";

import { chat } from "@tanstack/ai";
import { anthropicText } from "@tanstack/ai-anthropic";
import { trace } from "@opentelemetry/api";

import { openInferenceMiddleware } from "../src";

async function main() {
  const tracer = trace.getTracer("tanstack-ai-non-streaming-example");
  const text = await chat({
    adapter: anthropicText(
      (process.env["ANTHROPIC_MODEL"] as Parameters<typeof anthropicText>[0]) ?? "claude-haiku-4-5",
    ),
    stream: false,
    systemPrompts: [
      "You are a precise technical explainer. Answer directly, avoid marketing language, and keep the response concise.",
    ],
    messages: [
      {
        role: "user",
        content:
          "Write exactly two short sentences explaining what OpenInference is for. Do not use markdown or bullet points.",
      },
    ],
    middleware: [openInferenceMiddleware({ tracer })],
  });

  console.log("\nModel response:\n---------------");
  console.log(text);

  await new Promise((resolve) => setTimeout(resolve, 1500));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
