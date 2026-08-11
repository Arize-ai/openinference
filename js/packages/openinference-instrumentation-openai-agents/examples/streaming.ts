/**
 * Streaming example with a tool call.
 *
 * Run with:
 *   OPENAI_API_KEY=sk-... npx tsx examples/streaming.ts
 *
 * Demonstrates that AGENT, LLM, and TOOL spans are exported after a streamed
 * run is fully consumed.
 */
/* eslint-disable no-console */
import { tracerProvider } from "./instrumentation";

import { Agent, run, tool } from "@openai/agents";
import { z } from "zod";

const getWeather = tool({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: z.object({
    city: z.string().describe("City name"),
  }),
  execute: async ({ city }) =>
    JSON.stringify({ city, temperature: 22, unit: "Celsius", condition: "sunny" }),
});

const agent = new Agent({
  name: "StreamingWeatherAssistant",
  instructions:
    "You are a helpful assistant that provides weather information. " +
    "Use the get_weather tool when asked about weather.",
  tools: [getWeather],
});

async function main() {
  const result = await run(agent, "What is the weather in Tokyo?", { stream: true });

  console.log("\nStreamed output:");
  for await (const chunk of result.toTextStream({ compatibleWithNodeStreams: true })) {
    process.stdout.write(String(chunk));
  }

  await result.completed;
  console.log("\n\nFinal output:\n" + result.finalOutput);
}

main()
  .catch(console.error)
  .finally(async () => {
    await tracerProvider.forceFlush();
    await tracerProvider.shutdown();
  });
