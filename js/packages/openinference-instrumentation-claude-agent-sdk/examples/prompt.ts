/**
 * V2 prompt example — awaits a single prompt/result round-trip.
 * Produces a single `ClaudeAgent.prompt` AGENT span.
 *
 * Usage: npx tsx examples/prompt.ts
 */
/* eslint-disable no-console */
import "./instrumentation";

import { unstable_v2_prompt } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  const result = await unstable_v2_prompt(
    "Explain what OpenTelemetry is in one sentence.",
    {
      model: "claude-haiku-4-5",
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
    },
  );

  if (result.subtype === "success") {
    console.log("Result:", result.result);
  } else {
    console.error("Error:", result.subtype, result);
  }
}

main().catch(console.error);
