/**
 * V2 prompt-with-tools example — uses `unstable_v2_prompt` with allowed tools.
 * Produces a `ClaudeAgent.prompt` AGENT span with TOOL child spans for each
 * tool invocation.
 *
 * Usage: npx tsx examples/prompt-with-tools.ts
 */
/* eslint-disable no-console */
import "./instrumentation";

import { unstable_v2_prompt } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  const result = await unstable_v2_prompt(
    'Run `echo hello` in bash and tell me the output.',
    {
      model: "claude-haiku-4-5",
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
      allowedTools: ["Bash"],
    },
  );

  if (result.subtype === "success") {
    console.log("Result:", result.result);
  } else {
    console.error("Error:", result.subtype, result);
  }
}

main().catch(console.error);
