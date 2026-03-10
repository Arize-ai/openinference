/**
 * Simple V1 query example — sends a single prompt and streams the response.
 * Produces a single `ClaudeAgent.query` AGENT span.
 *
 * Usage: npx tsx examples/simple-query.ts
 */
/* eslint-disable no-console */
import "./instrumentation";

import { query } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  const messages = query({
    prompt: "What is 2 + 2?",
    options: {
      model: "claude-haiku-4-5",
      permissionMode: "bypassPermissions",
      allowDangerouslySkipPermissions: true,
      maxTurns: 1,
    },
  });

  for await (const message of messages) {
    console.log(`[${message.type}${message.subtype ? `:${message.subtype}` : ""}]`, JSON.stringify(message).slice(0, 200));
  }
}

main().catch(console.error);
