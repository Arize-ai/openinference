/**
 * V2 session example — demonstrates multi-turn conversation with tool use.
 * Each `send()` + `stream()` pair produces a `ClaudeAgent.turn` AGENT span
 * with TOOL child spans for each tool invocation.
 *
 * Usage: npx tsx examples/session.ts
 */
/* eslint-disable no-console */
import "./instrumentation";

import { unstable_v2_createSession } from "@anthropic-ai/claude-agent-sdk";

async function main() {
  const session = unstable_v2_createSession({
    model: "claude-haiku-4-5",
    permissionMode: "acceptEdits",
    allowedTools: ["Bash", "Read", "Write"],
  });

  // Turn 1: Create a file
  await session.send("Create a file called hello.txt with 'Hello, World!'");
  for await (const message of session.stream()) {
    console.log(`[Turn 1] [${message.type}${message.subtype ? `:${message.subtype}` : ""}]`, JSON.stringify(message).slice(0, 200));
  }

  console.log("Session ID:", session.sessionId);

  // Turn 2: Read it back
  await session.send("Read the file back and tell me what it says");
  for await (const message of session.stream()) {
    console.log(`[Turn 2] [${message.type}${message.subtype ? `:${message.subtype}` : ""}]`, JSON.stringify(message).slice(0, 200));
  }

  session.close();
  console.log("Session closed");
}

main().catch(console.error);
