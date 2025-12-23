/* eslint-disable no-console */
/**
 * Handoffs example for OpenAI Agents SDK instrumentation
 *
 * This example demonstrates how to:
 * 1. Set up OpenInference instrumentation for OpenAI Agents
 * 2. Create multiple specialized agents
 * 3. Use handoffs to transfer control between agents
 * 4. Track agent handoffs in traces (graph.node.id, graph.node.parent_id)
 *
 * Prerequisites:
 * - Set OPENAI_API_KEY environment variable
 * - Install dependencies: @openai/agents, @opentelemetry/sdk-trace-node
 */

import { instrumentation, provider } from "./instrumentation";

// IMPORTANT: Import the SDK as a namespace so we can pass it to instrument()
import * as agentsSdk from "@openai/agents";

// Create a Spanish translator agent
const spanishAgent = new agentsSdk.Agent({
  name: "SpanishTranslator",
  instructions:
    "You are a Spanish translator. Translate the user's message to Spanish. Only output the translation, nothing else.",
});

// Create a French translator agent
const frenchAgent = new agentsSdk.Agent({
  name: "FrenchTranslator",
  instructions:
    "You are a French translator. Translate the user's message to French. Only output the translation, nothing else.",
});

// Create the main triage agent that routes to specialized agents
// Use Agent.create for proper handoff type inference
const triageAgent = agentsSdk.Agent.create({
  name: "TriageAgent",
  instructions: `You are a helpful assistant that routes translation requests to the appropriate translator.

  - If the user wants to translate something to Spanish, hand off to the SpanishTranslator.
  - If the user wants to translate something to French, hand off to the FrenchTranslator.
  - For any other request, respond directly.`,
  handoffs: [spanishAgent, frenchAgent],
});

async function main() {
  // Instrument using the SDK module from our static import
  instrumentation.instrument(agentsSdk);

  console.log("Running handoffs example...\n");

  try {
    // Test Spanish handoff
    console.log("Request: \"Translate 'Hello, how are you?' to Spanish\"\n");
    const spanishResult = await agentsSdk.run(
      triageAgent,
      "Translate 'Hello, how are you?' to Spanish",
    );
    console.log("Spanish Translation:", spanishResult.finalOutput);
    console.log("\n---\n");

    // Test French handoff
    console.log("Request: \"Translate 'Good morning!' to French\"\n");
    const frenchResult = await agentsSdk.run(
      triageAgent,
      "Translate 'Good morning!' to French",
    );
    console.log("French Translation:", frenchResult.finalOutput);
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
