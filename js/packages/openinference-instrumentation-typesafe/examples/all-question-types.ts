import { choice, noul, score, TypeSafeClient } from "@typesafe-ai/sdk";

import { setupTracing } from "./instrumentation";

async function main() {
  const tracing = setupTracing("typesafe-all-question-types");
  try {
    const client = new TypeSafeClient();
    const result = await client.systemOne({
      state: { subject: "Payment failed", messages: ["I tried twice and need help today."] },
      questions: {
        category: choice({ task: "Choose the support team" }, {
          billing: { handles: ["payments", "refunds"] }, technical: "Product bugs", other: null,
        }),
        urgent: noul("Does this need immediate attention?"),
        severity: score("How severe is the issue?", ["Minor inconvenience", "Blocks a task", "Critical outage"]),
        // Plain question objects work alongside the builders.
        needs_reply: { type: "noul", instructions: "Does the customer need a reply?" },
      },
    });
    console.log(result);
  } finally {
    await tracing.shutdown();
  }
}

main().catch((error: unknown) => { console.error(error); process.exitCode = 1; });
