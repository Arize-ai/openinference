/* eslint-disable no-console */
/**
 * Streaming example for OpenAI Agents SDK instrumentation
 *
 * This example demonstrates how to:
 * 1. Set up OpenInference instrumentation for OpenAI Agents
 * 2. Use streaming to get real-time output from the agent
 * 3. Process streaming events as they arrive
 *
 * Prerequisites:
 * - Set OPENAI_API_KEY environment variable
 * - Install dependencies: @openai/agents, @opentelemetry/sdk-trace-node
 */

import { instrumentation, provider } from "./instrumentation";

// IMPORTANT: Import the SDK as a namespace so we can pass it to instrument()
import * as agentsSdk from "@openai/agents";

// Create a simple agent for streaming
const jokeAgent = new agentsSdk.Agent({
  name: "JokeAgent",
  instructions:
    "You are a comedian who tells funny jokes. When asked, tell exactly 3 short jokes, numbered 1-3.",
});

async function main() {
  // Instrument using the SDK module from our static import
  instrumentation.instrument(agentsSdk);

  console.log("Running streaming agent...\n");
  console.log("------- Streaming Output -------\n");

  try {
    // Run with streaming enabled
    const stream = await agentsSdk.run(jokeAgent, "Tell me some jokes!", { stream: true });

    // Process the text stream as it arrives
    for await (const chunk of stream.toTextStream()) {
      process.stdout.write(chunk);
    }

    console.log("\n\n------- End of Stream -------\n");

    // Get the final result after streaming completes
    const result = stream.finalOutput;
    console.log("Final output captured:", result ? "Yes" : "No");
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
