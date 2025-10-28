/* eslint-disable no-console */
import "./instrumentation";
import Anthropic from "@anthropic-ai/sdk";

async function toolUseExample() {
  const anthropic = new Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
  });

  // Define a simple tool
  const tools = [
    {
      name: "get_weather",
      description: "Get the current weather in a given location",
      input_schema: {
        type: "object" as const,
        properties: {
          location: {
            type: "string" as const,
            description: "The city and state, e.g. San Francisco, CA",
          },
        },
        required: ["location"],
      },
    },
  ];

  const message = await anthropic.messages.create({
    model: "claude-3-5-sonnet-latest",
    max_tokens: 1000,
    tools,
    messages: [
      {
        role: "user",
        content: "What is the weather like in San Francisco?",
      },
    ],
  });

  console.log("Response:", message.content);

  // Handle tool use in the response
  for (const content of message.content) {
    if (content.type === "tool_use") {
      console.log("Tool called:", content.name);
      console.log("Tool input:", content.input);

      // In a real application, you would call the actual tool here
      // and then send the result back to Claude
      const toolResult = {
        role: "user" as const,
        content: [
          {
            type: "tool_result" as const,
            tool_use_id: content.id,
            content: "The weather in San Francisco is sunny, 72°F",
          },
        ],
      };
      // Continue the conversation with the tool result
      const followUp = await anthropic.messages.create({
        model: "claude-3-5-sonnet-latest",
        max_tokens: 1000,
        tools,
        messages: [
          {
            role: "user",
            content: "What is the weather like in San Francisco?",
          },
          {
            role: "assistant",
            content: message.content,
          },
          toolResult,
        ],
      });

      console.log("Follow-up response:", followUp.content);
    }
  }
}

toolUseExample().catch(console.error);
