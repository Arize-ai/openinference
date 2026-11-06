/**
 * Image generation example.
 *
 * Run with:
 *   OPENAI_API_KEY=sk-... npx tsx examples/images.ts
 *
 * Demonstrates the AGENT -> LLM flow where the hosted image generation tool
 * result is recorded on the response span as `output.images.*`.
 */
/* eslint-disable no-console */
import { tracerProvider } from "./instrumentation";

import { Agent, imageGenerationTool, run } from "@openai/agents";

const agent = new Agent({
  name: "ImageAssistant",
  model: "gpt-4.1-mini",
  instructions: "You generate images when asked. Use the image generation tool.",
  tools: [
    imageGenerationTool({
      size: "1024x1024",
      // Compressed WebP keeps the recorded data URL small enough for Phoenix to render it.
      quality: "low",
      outputFormat: "webp",
      outputCompression: 0,
    }),
  ],
});

async function main() {
  const result = await run(agent, "Generate a small 1px white color dot");

  // The Agents SDK reports the hosted tool call under its Responses API item type.
  const images = result.output
    .filter((item) => item.type === "hosted_tool_call" && item.name === "image_generation_call")
    .map((item) => (item.type === "hosted_tool_call" ? item.output : undefined));
  console.log(`\nGenerated ${images.length} image(s): result=${Boolean(images[0])}`);
  console.log(`Recorded data URL length: ${(images[0]?.length ?? 0) + 24}`);

  if (result.finalOutput) {
    console.log("\nFinal output:\n" + result.finalOutput);
  }
}

main()
  .catch(console.error)
  .finally(async () => {
    await tracerProvider.forceFlush();
    await tracerProvider.shutdown();
  });
