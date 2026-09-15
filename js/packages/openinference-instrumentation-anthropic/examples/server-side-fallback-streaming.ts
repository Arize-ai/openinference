/* eslint-disable no-console */
import { shutdownTracing } from "./instrumentation";

import Anthropic from "@anthropic-ai/sdk";

const requestedModel = "claude-fable-5";
const fallbackModel = "claude-opus-4-8";

function createMidStreamFallbackFetch(): typeof fetch {
  const events: Array<Record<string, unknown> & { type: string }> = [
    {
      type: "message_start",
      message: {
        id: "msg_mid_stream_fallback_example",
        type: "message",
        role: "assistant",
        model: requestedModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        stop_details: null,
        usage: { input_tokens: 12, output_tokens: 1 },
      },
    },
    {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "Partial primary output." },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "content_block_start",
      index: 1,
      content_block: {
        type: "fallback",
        from: { model: requestedModel },
        to: { model: fallbackModel },
        trigger: { type: "refusal", category: "reasoning_extraction" },
      },
    },
    { type: "content_block_stop", index: 1 },
    {
      type: "content_block_start",
      index: 2,
      content_block: { type: "text", text: "" },
    },
    {
      type: "content_block_delta",
      index: 2,
      delta: { type: "text_delta", text: " Continued by the fallback model." },
    },
    { type: "content_block_stop", index: 2 },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null, stop_details: null },
      usage: {
        input_tokens: 12,
        output_tokens: 8,
        iterations: [
          { type: "message", model: requestedModel, input_tokens: 12, output_tokens: 3 },
          {
            type: "fallback_message",
            model: fallbackModel,
            input_tokens: 12,
            output_tokens: 5,
          },
        ],
      },
    },
    { type: "message_stop" },
  ];
  const body = events
    .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
    .join("");

  return async () =>
    new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

async function main() {
  try {
    const client = new Anthropic({
      apiKey: "fixture-api-key",
      fetch: createMidStreamFallbackFetch(),
    });
    const stream = await client.beta.messages.create({
      model: requestedModel,
      max_tokens: 128,
      messages: [{ role: "user", content: "Exercise a mid-stream fallback." }],
      stream: true,
      fallbacks: [{ model: fallbackModel }],
      betas: ["server-side-fallback-2026-07-01"],
    });

    const transitions: Array<{ from: string; to: string }> = [];
    for await (const event of stream) {
      if (event.type === "content_block_start" && event.content_block.type === "fallback") {
        transitions.push({
          from: event.content_block.from.model,
          to: event.content_block.to.model,
        });
      }
    }
    if (transitions.length !== 1 || transitions[0].to !== fallbackModel) {
      throw new Error(`Expected one fallback transition to ${fallbackModel}`);
    }

    console.log(
      JSON.stringify({ requestedModel, responseModel: fallbackModel, transitions }, null, 2),
    );
  } finally {
    await shutdownTracing();
  }
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
