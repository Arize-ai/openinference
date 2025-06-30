/* eslint-disable no-console */
import "./instrumentation";
import { isPatched } from "../src";
import OpenAI from "openai";
import { zodTextFormat } from "openai/helpers/zod";
import { z } from "zod";
import { Response as ResponseType } from "openai/resources/responses/responses";
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
  const stream = await openai.responses
    .create({
      instructions: "You are a helpful weather assistant.",
      input: [
        {
          type: "message",
          content: "What's the weather in Boston?",
          role: "user",
        },
      ],
      model: "gpt-4.1",
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
      stream: true,
    })
    .then(async (stream) => {
      let response: ResponseType | undefined;
      for await (const event of stream) {
        if (
          event.type === "response.output_item.added" &&
          event.item.type === "function_call"
        ) {
          console.log("function call\n----------");
          console.log(event.item.name);
        }
        if (event.type === "response.completed") {
          response = event.response;
        }
      }
      return response;
    });

  // respond to function call
  if (stream) {
    const id = stream.id;
    const fn = stream.output.find((o) => o.type === "function_call");
    if (fn) {
      const args = JSON.parse(fn.arguments);
      if (fn.name === "get_weather") {
        const weather = `The weather in ${args.location} is sunny with a temperature of 70 degrees.`;
        await openai.responses.create({
          previous_response_id: id,
          stream: false,
          input: [
            {
              type: "function_call_output",
              output: weather,
              call_id: fn.call_id,
            },
            {
              type: "message",
              content: "What should I wear?",
              role: "user",
            },
          ],
          model: "gpt-4.1",
        });
      }
    }
  }

  // built-in tools
  await openai.responses
    .create({
      model: "gpt-4.1",
      tools: [{ type: "web_search_preview" }],
      input: "What was a positive news story from today?",
    })
    .then((response) => {
      console.log(response.output_text);
    });

  // structured output
  const CalendarEvent = z.object({
    name: z.string(),
    date: z.string(),
    participants: z.array(z.string()),
  });
  const structured = await openai.responses.parse({
    model: "gpt-4.1",
    input: [
      {
        role: "system",
        content: "Extract the event information.",
      },
      {
        role: "user",
        content: "Alice and Bob are going to a science fair on Friday.",
      },
    ],
    text: {
      format: zodTextFormat(CalendarEvent, "event"),
    },
  });
  console.log(structured.output_parsed);
}

main();
