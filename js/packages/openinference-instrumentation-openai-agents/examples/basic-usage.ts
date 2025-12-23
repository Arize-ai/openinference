/* eslint-disable no-console */
/**
 * Basic usage example for OpenAI Agents SDK instrumentation
 *
 * This example demonstrates how to:
 * 1. Set up OpenInference instrumentation for OpenAI Agents
 * 2. Create a simple agent with a tool
 * 3. Run the agent and observe the generated spans
 *
 * Prerequisites:
 * - Set OPENAI_API_KEY environment variable
 * - Install dependencies: @openai/agents, @opentelemetry/sdk-trace-node
 */

import { instrumentation, provider } from "./instrumentation";

// IMPORTANT: Import the SDK as a namespace so we can pass it to instrument()
import * as agentsSdk from "@openai/agents";
import { z } from "zod";

// Define a simple weather tool
const getWeather = agentsSdk.tool({
  name: "get_weather",
  description: "Get the current weather for a location",
  parameters: z.object({
    location: z.string().describe("The city and state, e.g. San Francisco, CA"),
  }),
  execute: async ({ location }) => {
    // Mock weather data
    return {
      location,
      temperature: 72,
      unit: "F",
      conditions: "sunny",
    };
  },
});

// Create an agent with the weather tool
const weatherAgent = new agentsSdk.Agent({
  name: "WeatherAgent",
  instructions:
    "You are a helpful weather assistant. Use the get_weather tool to answer questions about the weather.",
  tools: [getWeather],
});

async function main() {
  // Instrument using the SDK module from our static import
  // This ensures the processor is registered with the correct module instance
  instrumentation.instrument(agentsSdk);

  console.log("Running weather agent...\n");

  try {
    const result = await agentsSdk.run(
      weatherAgent,
      "What's the weather like in San Francisco?",
    );

    console.log("\nAgent response:", result.finalOutput);
  } catch (error) {
    console.error("Error running agent:", error);
  }

  // Force flush spans to ensure they are exported
  await provider.forceFlush();

  // Give time for spans to be exported
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // Shutdown provider
  await provider.shutdown();
}

main().catch(console.error);
