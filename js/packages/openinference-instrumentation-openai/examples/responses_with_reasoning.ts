/* eslint-disable no-console */
import "./instrumentation";

import { isPatched } from "../src";

import OpenAI from "openai";
import type { Response as ResponseType } from "openai/resources/responses/responses";

if (!isPatched()) {
  throw new Error("OpenAI instrumentation failed");
}

const openai = new OpenAI();

function printReasoningSpanKeys(label: string, response: ResponseType) {
  console.log(`\n=== ${label} — span attribute verification ===`);
  console.log("Look for the following in the ConsoleSpanExporter output above:");
  response.output.forEach((item, i) => {
    if (item.type === "reasoning") {
      console.log(
        `  llm.output_messages.${i}.message.contents.0.message_content.type  = "reasoning"`,
      );
      console.log(
        `  llm.output_messages.${i}.message.contents.0.message_content.id    = "${item.id}"`,
      );
      const summaryText = item.summary.map((s) => s.text).join("\\n");
      console.log(
        `  llm.output_messages.${i}.message.contents.0.message_content.text  = "${summaryText}"`,
      );
      if (item.encrypted_content) {
        console.log(
          `  llm.output_messages.${i}.message.contents.0.message_content.encrypted_content = "<present>"`,
        );
      }
    }
  });
}

async function main() {
  // --- Non-streaming reasoning ---
  console.log("\n=== Non-streaming reasoning response ===");
  const response = await openai.responses.create({
    model: "o4-mini",
    input: "Write a one-sentence bedtime story about a unicorn.",
    reasoning: { effort: "low", summary: "auto" },
    include: ["reasoning.encrypted_content"],
  });

  response.output.forEach((item) => {
    if (item.type === "reasoning") {
      console.log("\n[reasoning block]");
      console.log("  id:", item.id);
      item.summary.forEach((s) => console.log("  summary:", s.text));
    } else if (item.type === "message") {
      item.content.forEach((part) => {
        if (part.type === "output_text") console.log("\n[answer]", part.text);
      });
    }
  });

  printReasoningSpanKeys("Non-streaming", response);

  // --- Streaming reasoning ---
  console.log("\n=== Streaming reasoning response ===\n");
  let streamedResponse: ResponseType | undefined;

  const stream = await openai.responses.create({
    model: "o4-mini",
    input: "Write a one-sentence bedtime story about a unicorn.",
    reasoning: { effort: "low", summary: "auto" },
    include: ["reasoning.encrypted_content"],
    stream: true,
  });

  for await (const event of stream) {
    if (event.type === "response.output_text.delta") {
      process.stdout.write(event.delta);
    }
    if (event.type === "response.completed") {
      streamedResponse = event.response;
    }
  }
  console.log();

  if (streamedResponse) {
    streamedResponse.output.forEach((item) => {
      if (item.type === "reasoning") {
        console.log("\n[reasoning block]");
        console.log("  id:", item.id);
        item.summary.forEach((s) => console.log("  summary:", s.text));
      }
    });
    printReasoningSpanKeys("Streaming", streamedResponse);
  }

  // --- Multi-turn: reasoning item passed back as input ---
  console.log("\n=== Multi-turn with reasoning in input ===");
  if (response.output.some((o) => o.type === "reasoning")) {
    const followUp = await openai.responses.create({
      model: "o4-mini",
      input: [
        // Pass the previous turn's output items back so the model can continue
        ...response.output.map((item) =>
          item.type === "reasoning"
            ? { type: "reasoning" as const, id: item.id, summary: item.summary }
            : {
                type: "message" as const,
                role: "assistant" as const,
                content: item.type === "message" ? item.content : [],
                status: "completed" as const,
              },
        ),
        { type: "message", role: "user", content: "Can you say it more simply?" },
      ],
      reasoning: { effort: "low", summary: "auto" },
    });

    followUp.output.forEach((item) => {
      if (item.type === "message") {
        item.content.forEach((part) => {
          if (part.type === "output_text") console.log("\n[follow-up answer]", part.text);
        });
      }
    });
    printReasoningSpanKeys("Multi-turn follow-up", followUp);
  }
}

main().catch(console.error);
