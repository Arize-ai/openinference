/* eslint-disable no-console */

import { tracerProvider } from "./instrumentation";

import { openai } from "@ai-sdk/openai";
import { LegacyOpenTelemetry } from "@ai-sdk/otel";
import { generateText } from "ai";

// LegacyOpenTelemetry emits the ai.* span shape, where usage arrives as ai.usage.* keys
// on the ai.generateText.doGenerate span instead of gen_ai.usage.*.
const legacyTelemetry = new LegacyOpenTelemetry();

// A reasoning model reports reasoning tokens, and OpenAI caches prompts of 1024+ tokens,
// so the second call with the same long prefix reports cached input tokens.
const model = openai(process.env["OPENAI_MODEL"] ?? "gpt-5-mini");

const longContext = Array.from(
  { length: 300 },
  (_, index) => `Fact ${index}: observability helps teams understand how AI applications behave.`,
).join("\n");

/**
 * Run a generateText call that reports total, reasoning and cached input token usage.
 * @param question the question to ask about the shared context
 */
async function ask(question: string) {
  const result = await generateText({
    model,
    system: longContext,
    prompt: question,
    telemetry: {
      functionId: "openinference-vercel-ai-sdk-usage-token-counts",
      integrations: legacyTelemetry,
    },
  });

  console.log(`\n${question}\n${result.text}`);
  console.log("usage:", JSON.stringify(result.usage));
}

/**
 * Run two calls with the same long prefix and wait for trace export.
 */
async function main() {
  await ask("How many facts are listed? Answer with a number only.");
  await ask("What is the last fact number? Answer with a number only.");

  console.log("\nDone.");

  await tracerProvider.forceFlush();
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
