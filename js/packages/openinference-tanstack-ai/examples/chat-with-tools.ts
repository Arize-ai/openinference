/* eslint-disable no-console */

import "./instrumentation";

import { chat, maxIterations, streamToText, toolDefinition } from "@tanstack/ai";
import { openaiText } from "@tanstack/ai-openai";
import { trace } from "@opentelemetry/api";
import { z } from "zod";

import { openInferenceMiddleware } from "../src";

const weatherTool = toolDefinition({
  name: "getWeather",
  description: "Get the weather for a city",
  inputSchema: z.object({ city: z.string() }),
  outputSchema: z.object({ forecast: z.string(), temperatureF: z.number() }),
}).server(async ({ city }) => {
  return {
    forecast: city === "Boston" ? "sunny" : "cloudy",
    temperatureF: city === "Boston" ? 70 : 65,
  };
});

async function main() {
  const tracer = trace.getTracer("tanstack-ai-example");
  const stream = chat({
    adapter: openaiText((process.env["OPENAI_MODEL"] as Parameters<typeof openaiText>[0]) ?? "gpt-4o-mini"),
    messages: [{ role: "user", content: "What is the weather in Boston? Use the tool." }],
    tools: [weatherTool],
    agentLoopStrategy: maxIterations(3),
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
