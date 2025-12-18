/* eslint-disable no-console */
/**
 * Lifecycle hooks example for OpenAI Agents SDK instrumentation
 *
 * This example demonstrates how to:
 * 1. Set up OpenInference instrumentation for OpenAI Agents
 * 2. Use agent.on() to listen for lifecycle events
 * 3. Track custom events during agent execution
 * 4. Monitor tool calls and responses
 *
 * Prerequisites:
 * - Set OPENAI_API_KEY environment variable
 * - Install dependencies: @openai/agents, @opentelemetry/sdk-trace-node
 */

import { instrumentation, provider } from "./instrumentation";

// IMPORTANT: Import the SDK as a namespace so we can pass it to instrument()
import * as agentsSdk from "@openai/agents";
import type { Usage, Agent } from "@openai/agents";
import { z } from "zod";

// Helper to format usage info
function toPrintableUsage(usage: Usage): string {
  if (!usage) return "No usage info";
  return (
    `${usage.requests ?? 0} requests, ` +
    `${usage.inputTokens ?? 0} input tokens, ` +
    `${usage.outputTokens ?? 0} output tokens, ` +
    `${usage.totalTokens ?? 0} total tokens`
  );
}

// Define tools for the agent
const searchTool = agentsSdk.tool({
  name: "search_database",
  description: "Search the internal database for information",
  parameters: z.object({
    query: z.string().describe("The search query"),
  }),
  execute: async ({ query }) => {
    // Simulate database search
    await new Promise((resolve) => setTimeout(resolve, 100));
    return {
      query,
      results: [
        { id: 1, title: `Result for "${query}" #1`, relevance: 0.95 },
        { id: 2, title: `Result for "${query}" #2`, relevance: 0.87 },
        { id: 3, title: `Result for "${query}" #3`, relevance: 0.72 },
      ],
      totalFound: 42,
    };
  },
});

const calculateTool = agentsSdk.tool({
  name: "calculate",
  description: "Perform mathematical calculations",
  parameters: z.object({
    expression: z.string().describe("Mathematical expression to evaluate"),
  }),
  execute: async ({ expression }) => {
    // Simple expression evaluation (in real app, use a proper parser)
    try {
      // Only allow safe mathematical operations
      const sanitized = expression.replace(/[^0-9+\-*/().]/g, "");
      const result = Function(`"use strict"; return (${sanitized})`)();
      return { expression, result, success: true };
    } catch {
      return { expression, error: "Invalid expression", success: false };
    }
  },
});

// Create the research agent
const researchAgent = new agentsSdk.Agent({
  name: "ResearchAssistant",
  instructions: `You are a helpful research assistant. You can:
  - Search the database for information using the search_database tool
  - Perform calculations using the calculate tool

  Use these tools to help answer user questions thoroughly.`,
  tools: [searchTool, calculateTool],
});

// Attach event hooks to the agent using the event emitter pattern
function attachHooks(agent: Agent<unknown, unknown>) {
  let eventCounter = 0;

  agent.on("agent_start", (ctx, currentAgent) => {
    eventCounter++;
    console.log(
      `\n🚀 [Hook ${eventCounter}] Agent "${currentAgent.name}" started`,
    );
    console.log(`   Usage: ${toPrintableUsage(ctx?.usage)}`);
  });

  agent.on("agent_end", (ctx, output) => {
    eventCounter++;
    console.log(`\n✅ [Hook ${eventCounter}] Agent "${agent.name}" ended`);
    console.log(`   Output type: ${typeof output}`);
    console.log(`   Usage: ${toPrintableUsage(ctx?.usage)}`);
  });

  agent.on("agent_tool_start", (ctx, currentTool, { toolCall }) => {
    eventCounter++;
    const args = toolCall.type === "function_call" ? toolCall.arguments : "";
    console.log(
      `\n🔧 [Hook ${eventCounter}] Tool "${currentTool.name}" started`,
    );
    console.log(`   Arguments: ${args}`);
    console.log(`   Usage: ${toPrintableUsage(ctx?.usage)}`);
  });

  agent.on("agent_tool_end", (ctx, currentTool, result, { toolCall }) => {
    eventCounter++;
    const args = toolCall.type === "function_call" ? toolCall.arguments : "";
    console.log(`\n✔️ [Hook ${eventCounter}] Tool "${currentTool.name}" ended`);
    console.log(`   Arguments: ${args}`);
    console.log(`   Result: ${JSON.stringify(result).slice(0, 100)}...`);
    console.log(`   Usage: ${toPrintableUsage(ctx?.usage)}`);
  });

  agent.on("agent_handoff", (ctx, nextAgent) => {
    eventCounter++;
    console.log(
      `\n🔄 [Hook ${eventCounter}] Handoff from "${agent.name}" to "${nextAgent.name}"`,
    );
    console.log(`   Usage: ${toPrintableUsage(ctx?.usage)}`);
  });
}

async function main() {
  // Instrument using the SDK module from our static import
  instrumentation.instrument(agentsSdk);

  // Attach hooks to the agent
  attachHooks(researchAgent);

  console.log("Running lifecycle hooks example...\n");
  console.log("Watch for [Hook] messages showing lifecycle events.\n");
  console.log("============================================\n");

  try {
    // Request that will trigger multiple tools
    console.log(
      'User: "Search for information about AI and calculate 42 * 17"\n',
    );

    const result = await agentsSdk.run(
      researchAgent,
      "Search for information about AI and calculate 42 * 17",
    );

    console.log("\n============================================\n");
    console.log("Final Response:", result.finalOutput);
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
