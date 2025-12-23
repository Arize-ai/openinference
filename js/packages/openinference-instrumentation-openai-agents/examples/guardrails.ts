/* eslint-disable no-console */
/**
 * Guardrails example for OpenAI Agents SDK instrumentation
 *
 * This example demonstrates how to:
 * 1. Set up OpenInference instrumentation for OpenAI Agents
 * 2. Create input guardrails to validate user input
 * 3. Create output guardrails to validate agent responses
 * 4. Track guardrail executions in traces
 *
 * Prerequisites:
 * - Set OPENAI_API_KEY environment variable
 * - Install dependencies: @openai/agents, @opentelemetry/sdk-trace-node
 */

import { instrumentation, provider } from "./instrumentation";

import type { InputGuardrail } from "@openai/agents";
// IMPORTANT: Import the SDK as a namespace so we can pass it to instrument()
import * as agentsSdk from "@openai/agents";

// Define an input guardrail that checks for inappropriate content
const contentFilter: InputGuardrail = {
  name: "ContentFilter",
  execute: async ({ input }) => {
    // Check for blocked words (simplified example)
    const blockedWords = ["spam", "hack", "illegal"];
    const inputStr = typeof input === "string" ? input : JSON.stringify(input);
    const lowerInput = inputStr.toLowerCase();

    for (const word of blockedWords) {
      if (lowerInput.includes(word)) {
        return {
          tripwireTriggered: true,
          outputInfo: {
            reason: `Input contains blocked word: ${word}`,
            blocked: true,
          },
        };
      }
    }

    return {
      tripwireTriggered: false,
      outputInfo: {
        reason: "Input passed content filter",
        blocked: false,
      },
    };
  },
};

// Define an input guardrail that limits input length
const lengthGuardrail: InputGuardrail = {
  name: "InputLengthGuardrail",
  execute: async ({ input }) => {
    const maxLength = 500;
    const inputStr = typeof input === "string" ? input : JSON.stringify(input);

    if (inputStr.length > maxLength) {
      return {
        tripwireTriggered: true,
        outputInfo: {
          reason: `Input exceeds maximum length of ${maxLength} characters`,
          blocked: true,
        },
      };
    }

    return {
      tripwireTriggered: false,
      outputInfo: {
        reason: "Input length is acceptable",
        blocked: false,
      },
    };
  },
};

// Create an agent with guardrails
const assistantAgent = new agentsSdk.Agent({
  name: "SafeAssistant",
  instructions:
    "You are a helpful assistant. Answer user questions concisely and helpfully.",
  inputGuardrails: [contentFilter, lengthGuardrail],
});

async function main() {
  // Instrument using the SDK module from our static import
  instrumentation.instrument(agentsSdk);

  console.log("Running guardrails example...\n");

  // Test 1: Normal input (should pass)
  console.log("Test 1: Normal input");
  console.log('Input: "What is the capital of France?"\n');
  try {
    const result1 = await agentsSdk.run(
      assistantAgent,
      "What is the capital of France?",
    );
    console.log("Response:", result1.finalOutput);
  } catch (error) {
    console.error("Guardrail triggered:", error);
  }
  console.log("\n---\n");

  // Test 2: Input with blocked word (should trigger guardrail)
  console.log("Test 2: Input with blocked word");
  console.log('Input: "How do I hack into a computer?"\n');
  try {
    const result2 = await agentsSdk.run(
      assistantAgent,
      "How do I hack into a computer?",
    );
    console.log("Response:", result2.finalOutput);
  } catch (error) {
    if (error instanceof agentsSdk.InputGuardrailTripwireTriggered) {
      console.log(
        "Guardrail triggered! The content filter blocked this request.",
      );
    } else if (error instanceof Error) {
      console.log("Error:", error.message);
    }
  }
  console.log("\n---\n");

  // Test 3: Another normal input (should pass)
  console.log("Test 3: Another normal input");
  console.log('Input: "Explain photosynthesis briefly"\n');
  try {
    const result3 = await agentsSdk.run(
      assistantAgent,
      "Explain photosynthesis briefly",
    );
    console.log("Response:", result3.finalOutput);
  } catch (error) {
    console.error("Error:", error);
  }

  // Force flush spans to ensure they are exported
  await provider.forceFlush();

  // Give time for spans to be exported
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // Shutdown provider
  await provider.shutdown();
}

main().catch(console.error);
