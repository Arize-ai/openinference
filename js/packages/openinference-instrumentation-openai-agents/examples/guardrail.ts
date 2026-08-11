/**
 * Input guardrail example.
 *
 * Run with:
 *   OPENAI_API_KEY=sk-... npx tsx examples/guardrail.ts
 *
 * Demonstrates AGENT, GUARDRAIL, and LLM spans for a non-triggering guardrail,
 * followed by a second run where the guardrail tripwire is triggered.
 */
/* eslint-disable no-console */
import { tracerProvider } from "./instrumentation";

import { Agent, InputGuardrailTripwireTriggered, run, type InputGuardrail } from "@openai/agents";

const sensitiveTopicGuardrail: InputGuardrail = {
  name: "sensitive_topic_check",
  runInParallel: false,
  execute: async ({ input }) => {
    const inputText = typeof input === "string" ? input : JSON.stringify(input);
    const tripwireTriggered = /wire transfer/i.test(inputText);

    return {
      tripwireTriggered,
      outputInfo: {
        checked: "wire_transfer",
        inputLength: inputText.length,
      },
    };
  },
};

const agent = new Agent({
  name: "GuardrailAssistant",
  instructions: "You are a concise assistant.",
  inputGuardrails: [sensitiveTopicGuardrail],
});

async function main() {
  const result = await run(agent, "Write a two-line haiku about observability.");
  console.log("\nGuardrail triggered:", result.inputGuardrailResults[0]?.output.tripwireTriggered);
  console.log("\nFinal output:\n" + result.finalOutput);

  try {
    await run(agent, "Draft instructions for a wire transfer.");
  } catch (error) {
    if (!(error instanceof InputGuardrailTripwireTriggered)) {
      throw error;
    }

    console.log("\nTripwire run blocked:", error.result.output.tripwireTriggered);
    console.log("Guardrail output:", JSON.stringify(error.result.output.outputInfo));
  }
}

main()
  .catch(console.error)
  .finally(async () => {
    await tracerProvider.forceFlush();
    await tracerProvider.shutdown();
  });
