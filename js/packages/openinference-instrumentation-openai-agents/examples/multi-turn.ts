/* eslint-disable no-console */
/**
 * Multi-turn conversation example for OpenAI Agents SDK instrumentation
 *
 * This example demonstrates how to:
 * 1. Set up OpenInference instrumentation for OpenAI Agents
 * 2. Maintain conversation history across multiple turns
 * 3. Use RunResult to continue conversations
 * 4. Track multi-turn conversations in traces
 *
 * Prerequisites:
 * - Set OPENAI_API_KEY environment variable
 * - Install dependencies: @openai/agents, @opentelemetry/sdk-trace-node
 */

import { instrumentation, provider } from "./instrumentation";

// IMPORTANT: Import the SDK as a namespace so we can pass it to instrument()
import * as agentsSdk from "@openai/agents";
import { z } from "zod";

// Create a memory tool to store and recall information
const memory: Record<string, string> = {};

const rememberTool = agentsSdk.tool({
  name: "remember",
  description: "Remember a piece of information with a given key",
  parameters: z.object({
    key: z.string().describe("The key to store the information under"),
    value: z.string().describe("The information to remember"),
  }),
  execute: async ({ key, value }) => {
    memory[key] = value;
    return `Remembered "${key}": "${value}"`;
  },
});

const recallTool = agentsSdk.tool({
  name: "recall",
  description: "Recall a piece of information by its key",
  parameters: z.object({
    key: z.string().describe("The key to recall"),
  }),
  execute: async ({ key }) => {
    const value = memory[key];
    if (value) {
      return `Recalled "${key}": "${value}"`;
    }
    return `No memory found for key: ${key}`;
  },
});

// Create an agent with memory capabilities
const memoryAgent = new agentsSdk.Agent({
  name: "MemoryAssistant",
  instructions: `You are a helpful assistant with memory capabilities.

  - Use the 'remember' tool to store information the user tells you
  - Use the 'recall' tool to retrieve previously stored information
  - Always confirm when you've remembered something
  - Help the user track and recall their information`,
  tools: [rememberTool, recallTool],
});

async function main() {
  // Instrument using the SDK module from our static import
  instrumentation.instrument(agentsSdk);

  console.log("Running multi-turn conversation example...\n");
  console.log("This example demonstrates maintaining conversation context.\n");

  try {
    // Turn 1: Store some information
    console.log("--- Turn 1 ---");
    console.log('User: "Remember that my favorite color is blue"\n');
    const turn1 = await agentsSdk.run(
      memoryAgent,
      "Remember that my favorite color is blue",
    );
    console.log("Assistant:", turn1.finalOutput);
    console.log("\n");

    // Turn 2: Store more information (continue from previous conversation)
    console.log("--- Turn 2 ---");
    console.log('User: "Also remember that my birthday is March 15"\n');
    const turn2 = await agentsSdk.run(
      memoryAgent,
      "Also remember that my birthday is March 15",
      {
        previousResponseId: turn1.lastResponseId,
      },
    );
    console.log("Assistant:", turn2.finalOutput);
    console.log("\n");

    // Turn 3: Recall information
    console.log("--- Turn 3 ---");
    console.log('User: "What is my favorite color?"\n');
    const turn3 = await agentsSdk.run(
      memoryAgent,
      "What is my favorite color?",
      {
        previousResponseId: turn2.lastResponseId,
      },
    );
    console.log("Assistant:", turn3.finalOutput);
    console.log("\n");

    // Turn 4: Ask about birthday
    console.log("--- Turn 4 ---");
    console.log('User: "When is my birthday?"\n');
    const turn4 = await agentsSdk.run(memoryAgent, "When is my birthday?", {
      previousResponseId: turn3.lastResponseId,
    });
    console.log("Assistant:", turn4.finalOutput);
    console.log("\n");

    // Turn 5: Test context understanding
    console.log("--- Turn 5 ---");
    console.log('User: "What have you remembered about me?"\n');
    const turn5 = await agentsSdk.run(
      memoryAgent,
      "What have you remembered about me? Use the recall tool if needed.",
      {
        previousResponseId: turn4.lastResponseId,
      },
    );
    console.log("Assistant:", turn5.finalOutput);
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
