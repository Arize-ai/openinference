/* eslint-disable no-console */

import "./instrumentation";

import { openai } from "@ai-sdk/openai";
import { stepCountIs, streamText, tool } from "ai";
import { z } from "zod";

const weatherTool = tool({
  description: "Get the weather in a location",
  inputSchema: z.object({
    location: z.string().describe("City and state, e.g. Boston, MA"),
  }),
  execute: async ({ location }) => {
    // Pretend we called a weather API.
    return {
      location,
      forecast: "sunny",
      temperatureF: 70,
    };
  },
});

async function main() {
  const result = streamText({
    model: openai(process.env["OPENAI_MODEL"] ?? "gpt-4o-mini"),
    tools: {
      weather: weatherTool,
    },
    // Allow the model to call a tool and then produce a final answer.
    stopWhen: stepCountIs(3),
    prompt:
      "What's the weather in Boston? Use the weather tool and then answer in one short sentence.",
    experimental_telemetry: {
      isEnabled: true,
      functionId: "openinference-vercel-ai-sdk-stream-tools",
      metadata: {
        example: "ai-sdk-v6",
        mode: "streamText+tools",
      },
    },
  });

  console.log("\n\nStreaming response:\n------------------");
  for await (const chunk of result.textStream) {
    process.stdout.write(chunk);
  }

  console.log("\n\nDone.");

  // Give exporters a moment to flush.
  await new Promise((resolve) => setTimeout(resolve, 1500));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
