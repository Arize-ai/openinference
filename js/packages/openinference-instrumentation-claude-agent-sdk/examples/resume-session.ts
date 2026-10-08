/**
 * V2 resume-session example — creates a session, performs a turn, closes it,
 * then resumes the same session with `unstable_v2_resumeSession`.
 *
 * Produces `ClaudeAgent.turn` AGENT spans for each send/stream pair across
 * both the original and resumed sessions.
 *
 * Usage: npx tsx examples/resume-session.ts
 */
/* eslint-disable no-console */
import "./instrumentation";

import {
  unstable_v2_createSession,
  unstable_v2_resumeSession,
} from "@anthropic-ai/claude-agent-sdk";

async function main() {
  // --- Original session ---
  const session = unstable_v2_createSession({
    model: "claude-haiku-4-5",
    permissionMode: "bypassPermissions",
    allowDangerouslySkipPermissions: true,
  });

  await session.send("Remember the number 42. Just acknowledge it.");
  for await (const message of session.stream()) {
    console.log(
      `[Turn 1] [${message.type}${message.subtype ? `:${message.subtype}` : ""}]`,
      JSON.stringify(message).slice(0, 200),
    );
  }

  const sessionId = session.sessionId;
  console.log("Session ID:", sessionId);

  session.close();
  console.log("Original session closed");

  // --- Resumed session ---
  const resumed = unstable_v2_resumeSession(sessionId, {
    model: "claude-haiku-4-5",
    permissionMode: "bypassPermissions",
    allowDangerouslySkipPermissions: true,
  });

  await resumed.send("What number did I ask you to remember?");
  for await (const message of resumed.stream()) {
    console.log(
      `[Turn 2 - resumed] [${message.type}${message.subtype ? `:${message.subtype}` : ""}]`,
      JSON.stringify(message).slice(0, 200),
    );
  }

  resumed.close();
  console.log("Resumed session closed");
}

main().catch(console.error);
