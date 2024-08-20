import { openai } from "@ai-sdk/openai";
import {
  embed,
  embedMany,
  generateObject,
  generateText,
  streamObject,
  streamText,
  tool,
} from "ai";
import { z } from "zod";

export async function POST(req: Request) {
  const { prompt } = await req.json();
  const textStream = await streamText({
    model: openai("gpt-3.5-turbo"),
    maxTokens: 100,
    messages: [
      { content: "hello, you are a great bot", role: "system" },
      { content: prompt, role: "user" },
    ],

    experimental_telemetry: {
      isEnabled: true,
      metadata: { example: "value" },
    },
  });

  await embed({
    model: openai.embedding("text-embedding-ada-002"),
    value: "hello, you are a great bot",
    experimental_telemetry: {
      isEnabled: true,
      metadata: { example: "value" },
    },
  });

  await embedMany({
    model: openai.embedding("text-embedding-ada-002"),
    values: ["hello, you are a great bot", prompt],
    experimental_telemetry: {
      isEnabled: true,
      metadata: { example: "value" },
    },
  });

  await generateObject({
    model: openai("gpt-3.5-turbo"),
    maxTokens: 100,
    messages: [
      { content: "hello, you are a great bot", role: "system" },
      { content: "please create me 3 objects like this", role: "user" },
    ],
    schema: z.object({
      description: z.string().describe("The generated text"),
      number: z.number().describe("A generated number"),
    }),
    experimental_telemetry: {
      isEnabled: true,
      metadata: { example: "value" },
    },
  });
  const objectStream = await streamObject({
    model: openai("gpt-3.5-turbo"),
    maxTokens: 100,
    messages: [
      { content: "hello, you are a great bot", role: "system" },
      { content: "please create me 3 objects like this", role: "user" },
    ],
    schema: z.object({
      description: z.string().describe("The generated text"),
      number: z.number().describe("A generated number"),
    }),
    experimental_telemetry: {
      isEnabled: true,
      metadata: { example: "value" },
    },
  });

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

  const textStreamResponse = await textStream.toTextStreamResponse().text();
  const objectStreamResponse = await objectStream.toTextStreamResponse().text();

  return new Response(
    JSON.stringify({
      text,
      toolResults,
      textStreamResponse,
      objectStreamResponse,
    }),
    {
      headers: { "Content-Type": "application/json" },
    },
  );
}
