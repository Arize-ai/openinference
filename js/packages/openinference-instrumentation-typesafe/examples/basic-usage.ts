import { choice, TypeSafeClient } from "@typesafe-ai/sdk";

import { setupTracing } from "./instrumentation";

async function main() {
  const tracing = setupTracing("typesafe-basic-usage");
  try {
    const client = new TypeSafeClient();
    const { data, requestId } = await client.systemOne({
      state: { document: "I was charged twice. Please fix this ASAP." },
      questions: {
        category: choice("What is this ticket about?", { billing: null, technical: null, other: null }),
      },
    }).withResponse();
    console.log({ requestId, category: data.answers.category });
  } finally {
    await tracing.shutdown();
  }
}

main().catch((error: unknown) => { console.error(error); process.exitCode = 1; });
