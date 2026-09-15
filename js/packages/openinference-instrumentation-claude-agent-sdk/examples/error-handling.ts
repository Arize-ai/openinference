/**
 * Error-handling example — triggers a budget error to demonstrate how
 * error results are captured in span attributes and status.
 *
 * Usage: npx tsx examples/error-handling.ts
 */
/* eslint-disable no-console */
import "./instrumentation";

import { unstable_v2_prompt } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  // Use an extremely low budget to trigger a budget error
  const result = await unstable_v2_prompt(
    "Write a 10,000 word essay about the history of computing.",
    {
      model: "claude-haiku-4-5",
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
      maxBudgetUsd: 0.0001,
    },
  );

  console.log("Result subtype:", result.subtype);

  if (result.subtype === "success") {
    console.log("Result:", result.result.slice(0, 200));
  } else {
    console.log("Error result:", JSON.stringify(result, null, 2));
  }
}

main().catch(console.error);
