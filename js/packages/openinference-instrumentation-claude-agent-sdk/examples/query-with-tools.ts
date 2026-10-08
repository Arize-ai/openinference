/**
 * V1 query example that triggers tool use (Bash).
 * Produces a `ClaudeAgent.query` AGENT span with TOOL child spans.
 *
 * Usage: npx tsx examples/query-with-tools.ts
 */
/* eslint-disable no-console */
import "./instrumentation";

import { query } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  const messages = query({
    prompt: "List the files in the current directory",
    options: {
      model: "claude-haiku-4-5",
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
      maxTurns: 3,
    },
  });

  for await (const message of messages) {
    console.log(`[${message.type}${message.subtype ? `:${message.subtype}` : ""}]`, JSON.stringify(message).slice(0, 200));
  }
}

main().catch(console.error);
