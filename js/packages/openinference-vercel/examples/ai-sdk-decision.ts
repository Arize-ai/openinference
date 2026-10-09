/* eslint-disable no-console */

import { createTypeSafeAi } from "@ai-sdk/typesafe-ai";
import { experimental_decide } from "ai";

import { tracerProvider } from "./instrumentation";

async function main() {
  const apiKey = process.env["TYPESAFE_API_KEY"];
  if (apiKey == null) {
    throw new Error("Set TYPESAFE_API_KEY before running the Jev example.");
  }

  const model = createTypeSafeAi({ apiKey }).decisionModel("jev-latest");
  const result = await experimental_decide({
    model,
    state: "Customer asks about an invoice.",
    questions: {
      route: {
        type: "choice",
        instructions: "Which team should handle this request?",
        criteria: { billing: "Invoices", support: "Product help" },
      },
      urgent: {
        type: "boolean",
        instructions: "Does this need immediate attention?",
      },
    },
    telemetry: { functionId: "openinference-vercel-decision" },
  });

  console.log("Jev model:", result.response.modelId);
  console.log("Decision answers:", result.answers);
  await tracerProvider.forceFlush();
}

main().catch((error: unknown) => {
  console.error(error);
  process.exitCode = 1;
});
