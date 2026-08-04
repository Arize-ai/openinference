/* eslint-disable no-console */

import "./instrumentation";

import { openai } from "@ai-sdk/openai";
import { generateText, streamText } from "ai";

const model = openai(process.env["OPENAI_MODEL"] ?? "gpt-4o-mini");

/**
 * Run a direct generateText call and emit its telemetry to Phoenix.
 */
async function runGenerateText() {
  // AI SDK v7 records generateText as an agent root span with a child chat span,
  // even when the call is made directly without an explicit agent abstraction.
  const result = await generateText({
    model,
    prompt: "Write one sentence about why observability matters for AI applications.",
    runtimeContext: {
      example: "ai-sdk-v7",
      mode: "generateText-direct",
    },
    telemetry: {
      functionId: "openinference-vercel-ai-sdk-generate-text-direct",
      includeRuntimeContext: {
        example: true,
        mode: true,
      },
    },
  });

  console.log("\n\ngenerateText response:\n----------------------");
  console.log(result.text);
}

/**
 * Run a direct streamText call and emit its telemetry to Phoenix.
 */
async function runStreamText() {
  // AI SDK v7 records streamText as an agent root span with a child chat span,
  // even when the call is made directly without an explicit agent abstraction.
  const result = streamText({
    model,
    prompt: "Write one sentence about tracing a streaming LLM response.",
    runtimeContext: {
      example: "ai-sdk-v7",
      mode: "streamText-direct",
    },
    telemetry: {
      functionId: "openinference-vercel-ai-sdk-stream-text-direct",
      includeRuntimeContext: {
        example: true,
        mode: true,
      },
    },
  });

  console.log("\n\nstreamText response:\n--------------------");
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }
  console.log();
}

/**
 * Run the direct AI SDK examples and wait briefly for trace export.
 */
async function main() {
  await runGenerateText();
  await runStreamText();

  console.log("\nDone.");

  // Give exporters a moment to flush.
  await new Promise((resolve) => setTimeout(resolve, 1500));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
