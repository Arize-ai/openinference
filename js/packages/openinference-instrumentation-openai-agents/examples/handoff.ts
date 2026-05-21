/**
 * Multi-agent handoff example.
 *
 * Run with:
 *   OPENAI_API_KEY=sk-... npx tsx examples/handoff.ts
 *
 * Demonstrates a handoff from a triage agent to a specialist agent.
 * In the printed spans you will see:
 *   - A `handoff to MathSpecialist` TOOL span recording the transfer
 *   - The MathSpecialist AGENT span carrying `graph.node.parent_id = TriageAgent`,
 *     making the multi-agent flow visualisable as a graph.
 */
/* eslint-disable no-console */
import "./instrumentation";

import { Agent, run } from "@openai/agents";

const mathAgent = new Agent({
  name: "MathSpecialist",
  instructions: "You are a math expert. Answer math questions concisely.",
});

const triageAgent = new Agent({
  name: "TriageAgent",
  instructions:
    "You MUST hand off math questions to the MathSpecialist. " +
    "Do not answer math questions yourself.",
  handoffs: [mathAgent],
});

async function main() {
  const result = await run(triageAgent, "What is 42 * 17?");
  console.log("\nFinal output:\n" + result.finalOutput);
}

main().catch(console.error);
