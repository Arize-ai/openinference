/* eslint-disable no-console */
import "./instrumentation";
import { isPatched } from "../src";
import OpenAI from "openai";

// Check if OpenAI has been patched
if (!isPatched()) {
  throw new Error("OpenAI instrumentation failed");
}

// Initialize OpenAI
const openai = new OpenAI();

async function main() {
  // non-streaming response
  await openai.responses
    .create({
      model: "gpt-4.1",
      input: "Write a one-sentence bedtime story about a unicorn.",
    })
    .then((response) => {
      console.log(response.output_text);
    });

  // streaming response
  await openai.responses
    .create({
      model: "gpt-4.1",
      input: "Get the weather in los angeles california",
      stream: true,
      tools: [
        {
          name: "get_weather",
          type: "function",
          parameters: {
            type: "object",
            properties: {
              location: { type: "string" },
            },
            additionalProperties: false,
            required: ["location"],
          },
          strict: true,
        },
      ],
    })
    .then(async (stream) => {
      for await (const event of stream) {
        if (
          event.type === "response.output_item.added" &&
          event.item.type === "function_call"
        ) {
          console.log("function call\n----------");
          console.log(event.item.name);
        }
      }
    });
}

main();
