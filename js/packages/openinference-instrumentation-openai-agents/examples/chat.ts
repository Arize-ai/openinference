/**
 * Single-agent example with a tool call.
 *
 * Run with:
 *   OPENAI_API_KEY=sk-... npx tsx examples/chat.ts
 *
 * Demonstrates the basic AGENT -> LLM -> TOOL -> LLM flow.
 */
/* eslint-disable no-console */
import "./instrumentation";

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
  name: "WeatherAssistant",
  instructions:
    "You are a helpful assistant that provides weather information. " +
    "Use the get_weather tool when asked about weather.",
  tools: [getWeather],
});

async function main() {
  const result = await run(agent, "What is the weather in Tokyo?");
  console.log("\nFinal output:\n" + result.finalOutput);
}

main().catch(console.error);
