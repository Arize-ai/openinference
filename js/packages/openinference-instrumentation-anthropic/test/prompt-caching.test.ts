import Anthropic from "@anthropic-ai/sdk";
import type { Attributes } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import { AnthropicInstrumentation } from "../src/instrumentation";
import { waitForSpans as waitForSpansOn } from "./helpers/waitForSpans";

const {
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
  LLM_TOKEN_COUNT_TOTAL,
} = SemanticConventions;

const TOKEN_COUNT_KEYS = [
  LLM_TOKEN_COUNT_PROMPT,
  LLM_TOKEN_COUNT_COMPLETION,
  LLM_TOKEN_COUNT_TOTAL,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ,
  LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE,
];

const model = "claude-sonnet-4-6";

type Usage = {
  input_tokens: number | null;
  output_tokens: number;
  cache_creation_input_tokens: number | null;
  cache_read_input_tokens: number | null;
};

const memoryExporter = new InMemorySpanExporter();
const waitForSpans = (count: number) => waitForSpansOn(memoryExporter, count);

function tokenCountAttributes(attributes: Attributes): Attributes {
  return Object.fromEntries(
    Object.entries(attributes).filter(([key]) => TOKEN_COUNT_KEYS.includes(key)),
  );
}

function createJSONFetch(usage: Usage): typeof fetch {
  return async () =>
    new Response(
      JSON.stringify({
        id: "msg_cache",
        type: "message",
        role: "assistant",
        model,
        content: [{ type: "text", text: "Cached response" }],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage,
      }),
      { status: 200, headers: { "content-type": "application/json" } },
    );
}

function createStreamingFetch({
  startUsage,
  deltaUsage,
}: {
  startUsage: Usage;
  deltaUsage: Usage;
}): typeof fetch {
  const events: Array<Record<string, unknown> & { type: string }> = [
    {
      type: "message_start",
      message: {
        id: "msg_cache_stream",
        type: "message",
        role: "assistant",
        model,
        usage: startUsage,
        content: [],
        stop_reason: null,
        stop_sequence: null,
      },
    },
    { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
    {
      type: "content_block_delta",
      index: 0,
      delta: { type: "text_delta", text: "Cached response" },
    },
    { type: "content_block_stop", index: 0 },
    {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: deltaUsage,
    },
    { type: "message_stop" },
  ];
  const body = events
    .map((event) => `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`)
    .join("");

  return async () =>
    new Response(body, { status: 200, headers: { "content-type": "text/event-stream" } });
}

// input=10, cache write=1733, cache read=512, output=5 → prompt 2255, total 2260.
const cases: Array<{ name: string; usage: Usage; expected: Attributes }> = [
  {
    name: "cache write and read",
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: 1733,
      cache_read_input_tokens: 512,
    },
    expected: {
      [LLM_TOKEN_COUNT_PROMPT]: 2255,
      [LLM_TOKEN_COUNT_COMPLETION]: 5,
      [LLM_TOKEN_COUNT_TOTAL]: 2260,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: 1733,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: 512,
    },
  },
  {
    name: "cache write only",
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: 1733,
      cache_read_input_tokens: 0,
    },
    expected: {
      [LLM_TOKEN_COUNT_PROMPT]: 1743,
      [LLM_TOKEN_COUNT_COMPLETION]: 5,
      [LLM_TOKEN_COUNT_TOTAL]: 1748,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_WRITE]: 1733,
    },
  },
  {
    name: "cache read only",
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 512,
    },
    expected: {
      [LLM_TOKEN_COUNT_PROMPT]: 522,
      [LLM_TOKEN_COUNT_COMPLETION]: 5,
      [LLM_TOKEN_COUNT_TOTAL]: 527,
      [LLM_TOKEN_COUNT_PROMPT_DETAILS_CACHE_READ]: 512,
    },
  },
  {
    name: "zero cache counts",
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: 0,
    },
    expected: {
      [LLM_TOKEN_COUNT_PROMPT]: 10,
      [LLM_TOKEN_COUNT_COMPLETION]: 5,
      [LLM_TOKEN_COUNT_TOTAL]: 15,
    },
  },
  {
    name: "null cache counts",
    usage: {
      input_tokens: 10,
      output_tokens: 5,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null,
    },
    expected: {
      [LLM_TOKEN_COUNT_PROMPT]: 10,
      [LLM_TOKEN_COUNT_COMPLETION]: 5,
      [LLM_TOKEN_COUNT_TOTAL]: 15,
    },
  },
];

describe("AnthropicInstrumentation - prompt caching token counts", () => {
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

  async function createMessage(client: Anthropic) {
    await client.messages.create({
      model,
      max_tokens: 100,
      messages: [{ role: "user", content: "Hello" }],
    });
    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    return tokenCountAttributes(spans[0].attributes);
  }

  async function streamMessage(client: Anthropic) {
    const stream = await client.messages.create({
      model,
      max_tokens: 100,
      messages: [{ role: "user", content: "Hello" }],
      stream: true,
    });
    for await (const _event of stream) {
      // Drain the caller's side of the tee'd stream.
    }
    await waitForSpans(1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    return tokenCountAttributes(spans[0].attributes);
  }

  it.each(cases)("records $name for non-streaming responses", async ({ usage, expected }) => {
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: createJSONFetch(usage) });

    expect(await createMessage(client)).toEqual(expected);
  });

  it.each(cases)("records $name for streaming responses", async ({ usage, expected }) => {
    // message_start carries the input side and a placeholder output count;
    // message_delta repeats the cumulative counts with the final output.
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: createStreamingFetch({
        startUsage: { ...usage, output_tokens: 1 },
        deltaUsage: usage,
      }),
    });

    expect(await streamMessage(client)).toEqual(expected);
  });

  it("keeps message_start cache counts when message_delta omits them", async () => {
    const [{ usage, expected }] = cases;
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: createStreamingFetch({
        startUsage: { ...usage, output_tokens: 1 },
        deltaUsage: {
          input_tokens: null,
          output_tokens: 5,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
        },
      }),
    });

    expect(await streamMessage(client)).toEqual(expected);
  });

  it("includes message_start cache counts when message_delta reports only input tokens", async () => {
    const [{ usage, expected }] = cases;
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: createStreamingFetch({
        startUsage: { ...usage, output_tokens: 1 },
        deltaUsage: {
          input_tokens: 10,
          output_tokens: 5,
          cache_creation_input_tokens: null,
          cache_read_input_tokens: null,
        },
      }),
    });

    // The prompt count must never be smaller than its own cache parts.
    expect(await streamMessage(client)).toEqual(expected);
  });
});
