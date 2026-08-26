import Anthropic from "@anthropic-ai/sdk";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";

const { LLM_MODEL_NAME, LLM_REQUEST_MODEL_NAME, LLM_RESPONSE_MODEL_NAME } = SemanticConventions;

const requestedModel = "claude-fable-5";
const fallbackModel = "claude-opus-4-8";

const memoryExporter = new InMemorySpanExporter();

async function waitForSpans(count: number) {
  for (let i = 0; i < 50; i++) {
    if (memoryExporter.getFinishedSpans().length >= count) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
}

function createJSONFetch(): typeof fetch {
  return async () =>
    new Response(
      JSON.stringify({
        id: "msg_fallback",
        type: "message",
        role: "assistant",
        model: fallbackModel,
        content: [
          {
            type: "fallback",
            from: { model: requestedModel },
            to: { model: fallbackModel },
            trigger: { type: "refusal", category: "cyber" },
          },
          { type: "text", text: "Fallback response" },
        ],
        stop_reason: "end_turn",
        stop_sequence: null,
        stop_details: null,
        usage: { input_tokens: 12, output_tokens: 4 },
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
}

function createStreamingFetch(): typeof fetch {
  const events: Array<Record<string, unknown> & { type: string }> = [
    {
      type: "message_start",
      message: {
        id: "msg_fallback_stream",
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
      delta: { type: "text_delta", text: "Partial response" },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "content_block_start",
      index: 1,
      content_block: {
        type: "fallback",
        from: { model: requestedModel },
        to: { model: fallbackModel },
        trigger: { type: "refusal", category: "cyber" },
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
      delta: { type: "text_delta", text: " served by fallback" },
    },
    { type: "content_block_stop", index: 2 },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null, stop_details: null },
      usage: {
        input_tokens: 12,
        output_tokens: 6,
        iterations: [
          { type: "message", model: requestedModel, input_tokens: 12, output_tokens: 2 },
          {
            type: "fallback_message",
            model: fallbackModel,
            input_tokens: 12,
            output_tokens: 4,
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

describe("AnthropicInstrumentation - server-side fallback", () => {
  const tracerProvider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
  });
  tracerProvider.register();
  const instrumentation = new AnthropicInstrumentation({ tracerProvider });
  instrumentation.disable();
  instrumentation._modules[0].moduleExports = Anthropic;

  beforeAll(() => {
    instrumentation.enable();
  });

  beforeEach(() => {
    memoryExporter.reset();
  });

  afterEach(() => {
    instrumentation.disable();
    instrumentation.enable();
  });

  it("instruments beta.messages.create", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: createJSONFetch() });

    await client.beta.messages.create({
      model: requestedModel,
      max_tokens: 100,
      messages: [{ role: "user", content: "Hello" }],
      fallbacks: "default",
      betas: ["server-side-fallback-2026-07-01"],
    });

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes[LLM_REQUEST_MODEL_NAME]).toBe(requestedModel);
    expect(spans[0].attributes[LLM_RESPONSE_MODEL_NAME]).toBe(fallbackModel);
    expect(spans[0].attributes[LLM_MODEL_NAME]).toBe(fallbackModel);
  });

  it("updates the response model at a mid-stream fallback boundary", async () => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: createStreamingFetch() });

    const stream = await client.beta.messages.create({
      model: requestedModel,
      max_tokens: 100,
      messages: [{ role: "user", content: "Hello" }],
      stream: true,
      fallbacks: "default",
      betas: ["server-side-fallback-2026-07-01"],
    });
    for await (const _event of stream) {
      // Drain the caller's side of the tee'd stream.
    }

    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0].attributes[LLM_REQUEST_MODEL_NAME]).toBe(requestedModel);
    expect(spans[0].attributes[LLM_RESPONSE_MODEL_NAME]).toBe(fallbackModel);
    expect(spans[0].attributes[LLM_MODEL_NAME]).toBe(fallbackModel);
  });
});
