/* eslint-disable no-console */
import "./instrumentation";

import { GoogleGenAI } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY!,
});

async function main() {
  console.log("Function calling example:\n");

  const tools = [
    {
      functionDeclarations: [
        {
          name: "get_weather",
          description: "Get the current weather for a location",
          parametersJsonSchema: {
            type: "object",
            properties: {
              location: {
                type: "string",
                description: "The city and state, e.g. San Francisco, CA",
              },
              unit: {
                type: "string",
                enum: ["celsius", "fahrenheit"],
                description: "The temperature unit",
              },
            },
            required: ["location"],
          },
        },
      ],
    },
  ];

  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "What's the weather like in San Francisco?",
    config: {
      tools,
    },
  });

  if (response.functionCalls && response.functionCalls.length > 0) {
    console.log("Function call requested:");
    console.log("Name:", response.functionCalls[0].name);
    console.log(
      "Arguments:",
      JSON.stringify(response.functionCalls[0].args, null, 2),
    );

    // Send the tool result back to the model so it can produce a final answer.
    const responseWithResult = await ai.models.generateContent({
      model: "gemini-2.5-flash",
      contents: [
        {
          role: "user",
          parts: [{ text: "What's the weather like in San Francisco?" }],
        },
        {
          role: "model",
          parts: [
            {
              functionCall: {
                name: "get_weather",
                args: response.functionCalls[0].args,
              },
            },
          ],
        },
        {
          role: "user",
          parts: [
            {
              functionResponse: {
                name: response.functionCalls[0].name,
                response: {
                  output: { text: "The weather in San Francisco is sunny." },
                },
              },
            },
          ],
        },
      ],
    });

    console.log("\nFinal response:", responseWithResult.text);
  } else {
    console.log("Response:", response.text);
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
