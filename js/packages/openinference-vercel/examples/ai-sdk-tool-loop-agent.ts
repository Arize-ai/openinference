/* eslint-disable no-console */

import "./instrumentation";

import { openai } from "@ai-sdk/openai";
import { stepCountIs, tool, ToolLoopAgent } from "ai";
import { z } from "zod";

const weatherTool = tool({
  description: "Get the weather in a location",
  inputSchema: z.object({
    location: z.string().describe("City and state, e.g. Boston, MA"),
  }),
  execute: async ({ location }) => {
    return {
      location,
      forecast: "sunny",
      temperatureF: 70,
    };
  },
});

const calculatorTool = tool({
  description: "Evaluate a basic arithmetic expression",
  inputSchema: z.object({
    expression: z
      .string()
      .describe("Arithmetic expression like '100 * 25 + 3'"),
  }),
  execute: async ({ expression }) => {
    // Extremely limited + intentionally strict for an example.
    // Supports digits, whitespace, + - * / and parentheses.
    if (!/^[0-9+\-*/()\s.]+$/.test(expression)) {
      throw new Error("Expression contains unsupported characters");
    }

    const value = Function(`"use strict"; return (${expression});`)() as number;
    return { expression, value };
  },
});

const agent = new ToolLoopAgent({
  model: openai(process.env["OPENAI_MODEL"] ?? "gpt-4o-mini"),
  instructions:
    "You are a helpful assistant. Use tools when they provide more reliable answers.",
  tools: {
    weather: weatherTool,
    calculator: calculatorTool,
  },
  stopWhen: stepCountIs(6),
  experimental_telemetry: {
    isEnabled: true,
    functionId: "openinference-vercel-ai-sdk-tool-loop-agent",
    metadata: {
      example: "ai-sdk-v6",
      primitive: "ToolLoopAgent",
    },
  },
});

async function main() {
  const stream = await agent.stream({
    prompt:
      "What's the weather in Boston, and what is 100 * 25 + 3? Use the tools and then answer in 2 short sentences.",
  });

  console.log("\n\nAgent stream:\n------------");
  for await (const chunk of stream.textStream) {
    process.stdout.write(chunk);
  }

  console.log("\n\nDone.");
  await new Promise((resolve) => setTimeout(resolve, 1500));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
