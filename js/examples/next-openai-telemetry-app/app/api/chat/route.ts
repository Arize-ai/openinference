import { openai } from "@ai-sdk/openai";
import { streamText } from "ai";

export async function POST(req: Request) {
  const { messages } = await req.json();
  const textStream = await streamText({
    model: openai("gpt-3.5-turbo"),
    maxTokens: 100,
    messages: messages,
    experimental_telemetry: {
      isEnabled: true,
      metadata: { route: "api/chat" },
    },
  });

  return textStream.toDataStreamResponse();
}
