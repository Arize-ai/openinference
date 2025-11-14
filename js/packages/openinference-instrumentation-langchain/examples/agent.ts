import "./instrumentation";
import "dotenv/config";

import { createAgent, tool } from "langchain";
import * as z from "zod";

const main = async () => {
  // Create tools that the agent can use
  const getWeather = tool(
    ({ city }) => {
      // Simulate weather API call
      const weatherConditions = [
        "sunny",
        "cloudy",
        "rainy",
        "snowy",
        "partly cloudy",
      ];
      const temperature = Math.floor(Math.random() * 40) + 10; // 10-50°C
      const condition =
        weatherConditions[Math.floor(Math.random() * weatherConditions.length)];

      return `The weather in ${city} is currently ${condition} with a temperature of ${temperature}°C.`;
    },
    {
      name: "get_weather",
      description: "Get the current weather for a given city",
      schema: z.object({
        city: z.string().describe("The name of the city to get weather for"),
      }),
    },
  );

  const calculateMath = tool(
    ({ expression }) => {
      try {
        // Simple math evaluation (in production, use a safer math parser)
        const result = Function(`"use strict"; return (${expression})`)();
        return `The result of ${expression} is ${result}`;
      } catch {
        return `Error calculating ${expression}: Invalid mathematical expression`;
      }
    },
    {
      name: "calculate_math",
      description: "Calculate mathematical expressions",
      schema: z.object({
        expression: z
          .string()
          .describe(
            "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5')",
          ),
      }),
    },
  );

  const searchInfo = tool(
    ({ query }) => {
      // Simulate a knowledge base search
      const knowledgeBase = {
        langchain:
          "LangChain is a framework for developing applications powered by language models. It provides tools for building agents, chains, and more.",
        openai:
          "OpenAI is an AI research company that created GPT models and provides API access to various language models.",
        javascript:
          "JavaScript is a programming language commonly used for web development, both frontend and backend applications.",
        typescript:
          "TypeScript is a strongly typed programming language that builds on JavaScript by adding static type definitions.",
        ai: "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn.",
      };

      const lowerQuery = query.toLowerCase();
      for (const [key, value] of Object.entries(knowledgeBase)) {
        if (lowerQuery.includes(key)) {
          return `Information about ${key}: ${value}`;
        }
      }

      return `I don't have specific information about "${query}" in my knowledge base. Try asking about LangChain, OpenAI, JavaScript, TypeScript, or AI.`;
    },
    {
      name: "search_info",
      description: "Search for information in the knowledge base",
      schema: z.object({
        query: z.string().describe("The search query to look up information"),
      }),
    },
  );

  // Create the agent with multiple tools
  const agent = createAgent({
    model: "gpt-4o-mini", // Using a more accessible model
    tools: [getWeather, calculateMath, searchInfo],
  });

  // eslint-disable-next-line no-console
  console.log(
    "🤖 Agent created with tools: weather, math calculator, and info search",
  );
  // eslint-disable-next-line no-console
  console.log(
    "📊 OpenInference tracing is active - check your Phoenix dashboard!",
  );

  // Example conversations to demonstrate different tools
  const conversations = [
    "What's the weather like in Tokyo?",
    "Can you calculate 15 * 7 + 23?",
    "Tell me about LangChain",
    "What's the weather in Paris and can you also calculate 100 / 4?",
  ];

  for (const [index, message] of conversations.entries()) {
    // eslint-disable-next-line no-console
    console.log(`\n--- Conversation ${index + 1} ---`);
    // eslint-disable-next-line no-console
    console.log(`User: ${message}`);

    try {
      const response = await agent.invoke({
        messages: [{ role: "user", content: message }],
      });

      // Extract the final message content
      const finalMessage = response.messages[response.messages.length - 1];
      // eslint-disable-next-line no-console
      console.log(`Agent: ${finalMessage.content}`);
    } catch (error) {
      // eslint-disable-next-line no-console
      console.error(`Error in conversation ${index + 1}:`, error);
    }
  }

  // eslint-disable-next-line no-console
  console.log("\n✅ Agent example completed!");
  // eslint-disable-next-line no-console
  console.log(
    "🔍 Check your Phoenix dashboard at http://localhost:6006 to see the traces",
  );
};

// Run the example
// eslint-disable-next-line no-console
main().catch(console.error);
