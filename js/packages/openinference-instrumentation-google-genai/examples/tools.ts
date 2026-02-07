/* eslint-disable no-console */
import "./instrumentation";

import { createInstrumentedGoogleGenAI } from "../src";

// Create an instrumented GoogleGenAI instance
const ai = createInstrumentedGoogleGenAI({
  apiKey: process.env.GOOGLE_API_KEY || process.env.GEMINI_API_KEY!,
});

async function main() {
  console.log("Function calling example:\n");

  // Define a tool/function
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

  // Make a request with tool definitions
  const response = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: "What's the weather like in San Francisco?",
    config: {
      tools,
    },
  });

  // Check if the model wants to call a function
  if (response.functionCalls && response.functionCalls.length > 0) {
    console.log("Function call requested:");
    console.log("Name:", response.functionCalls[0].name);
    console.log(
      "Arguments:",
      JSON.stringify(response.functionCalls[0].args, null, 2),
    );

    // In a real application, you would:
    // 1. Execute the function with the provided arguments
    // 2. Send the result back to the model for a final response
    console.log(
      "\n--- In production, you would execute the function and send results back ---",
    );
  } else {
    console.log("Response:", response.text);
  }
}

main().catch((error) => {
  console.error("Error:", error);
  process.exit(1);
});
