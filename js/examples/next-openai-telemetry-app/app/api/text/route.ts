import { openai } from "@ai-sdk/openai";
import { generateText, tool } from "ai";
import { z } from "zod";

export async function POST(req: Request) {
  const { prompt } = await req.json();

  const { text, toolResults } = await generateText({
    model: openai("gpt-3.5-turbo"),
    maxTokens: 100,
    messages: [
      { content: "hello, you are a great bot", role: "system" },
      { content: "what is the weather in kentucky", role: "user" },
    ],
    tools: {
      weather: tool({
        parameters: z.object({
          location: z.string().describe("The location to get the weather for"),
        }),
        execute: async ({ location }) => ({
          location,
          temperature: 72 + Math.floor(Math.random() * 21) - 10,
        }),
      }),
    },
    experimental_telemetry: {
      isEnabled: true,
      functionId: "example-function-id",
      metadata: { example: "value" },
    },
  });

  return new Response(JSON.stringify({ text, toolResults }), {
    headers: { "Content-Type": "application/json" },
  });
}
