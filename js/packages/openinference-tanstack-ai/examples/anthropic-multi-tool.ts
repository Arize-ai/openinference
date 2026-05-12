/* eslint-disable no-console */

import "./instrumentation";

import { chat, maxIterations, streamToText, toolDefinition } from "@tanstack/ai";
import { anthropicText } from "@tanstack/ai-anthropic";
import { trace } from "@opentelemetry/api";
import { z } from "zod";

import { openInferenceMiddleware } from "../src";

const weatherByCity: Record<string, { forecast: string; temperatureF: number }> = {
  boston: { forecast: "sunny", temperatureF: 70 },
  seattle: { forecast: "misty", temperatureF: 58 },
};

const timeByCity: Record<string, string> = {
  boston: "2026-04-08T18:30:00-04:00",
  seattle: "2026-04-08T15:30:00-07:00",
};

const weatherTool = toolDefinition({
  name: "getWeather",
  description: "Get the weather for a city",
  inputSchema: z.object({ city: z.string() }),
  outputSchema: z.object({ forecast: z.string(), temperatureF: z.number() }),
}).server(async ({ city }) => {
  const normalizedCity = city.trim().toLowerCase();
  return weatherByCity[normalizedCity] ?? { forecast: "unknown", temperatureF: 65 };
});

const localTimeTool = toolDefinition({
  name: "getLocalTime",
  description: "Get the current local time for a city",
  inputSchema: z.object({ city: z.string() }),
  outputSchema: z.object({ isoTime: z.string() }),
}).server(async ({ city }) => {
  const normalizedCity = city.trim().toLowerCase();
  return { isoTime: timeByCity[normalizedCity] ?? "2026-04-08T12:00:00Z" };
});

async function main() {
  const tracer = trace.getTracer("tanstack-ai-anthropic-example");
  const stream = chat({
    adapter: anthropicText(
      (process.env["ANTHROPIC_MODEL"] as Parameters<typeof anthropicText>[0]) ?? "claude-haiku-4-5",
    ),
    messages: [
      {
        role: "user",
        content:
          "Compare Boston and Seattle. You must call getWeather and getLocalTime for both cities before answering. Return a short markdown table followed by one sentence of analysis.",
      },
    ],
    tools: [weatherTool, localTimeTool],
    agentLoopStrategy: maxIterations(6),
    middleware: [openInferenceMiddleware({ tracer })],
  });

  const text = await streamToText(stream);
  console.log("\nModel response:\n---------------");
  console.log(text);

  await new Promise((resolve) => setTimeout(resolve, 1500));
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
