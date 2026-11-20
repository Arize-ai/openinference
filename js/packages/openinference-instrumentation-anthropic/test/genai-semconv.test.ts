import Anthropic from "@anthropic-ai/sdk";
import type { Attributes } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { TraceConfigOptions } from "@arizeai/openinference-core";

import { AnthropicInstrumentation } from "../src/instrumentation";
import { jsonFetch, sseFetch, textStreamEvents } from "./helpers/sse";
import { waitForSpans as waitForSpansOn } from "./helpers/waitForSpans";

const GEN_AI_PREFIX = "gen_ai.";

/** `gen_ai.*` attributes whose values are JSON encoded on the span. */
const JSON_ENCODED_ATTRIBUTES = [
  "gen_ai.input.messages",
  "gen_ai.output.messages",
  "gen_ai.tool.definitions",
];

/**
 * Picks the `gen_ai.*` attributes of a span, decoding the JSON encoded ones so
 * a snapshot shows their structure.
 * @param attributes - The attributes of an exported span
 * @returns The `gen_ai.*` attributes, keyed by attribute name
 */
function getGenAIAttributes(attributes: Attributes): Record<string, unknown> {
  const genAIAttributes: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(attributes)) {
    if (!key.startsWith(GEN_AI_PREFIX)) {
      continue;
    }
    const isJSONEncoded = JSON_ENCODED_ATTRIBUTES.includes(key) && typeof value === "string";
    genAIAttributes[key] = isJSONEncoded ? JSON.parse(value) : value;
  }
  return genAIAttributes;
}

const memoryExporter = new InMemorySpanExporter();
const tracerProvider = new NodeTracerProvider({
  spanProcessors: [new SimpleSpanProcessor(memoryExporter)],
});
tracerProvider.register();

const toolUseMessage = {
  id: "msg_tool_use",
  type: "message",
  role: "assistant",
  model: "claude-sonnet-4-6",
  content: [
    { type: "text", text: "Let me check the weather." },
    {
      type: "tool_use",
      id: "toolu_weather",
      name: "get_weather",
      input: { location: "Boston" },
    },
  ],
  stop_reason: "tool_use",
  stop_sequence: null,
  usage: {
    input_tokens: 40,
    output_tokens: 12,
    cache_read_input_tokens: 16,
    cache_creation_input_tokens: 8,
  },
};

const toolUseRequest = {
  model: "claude-sonnet-4-6",
  max_tokens: 256,
  temperature: 0.2,
  system: "You are a weather assistant.",
  messages: [{ role: "user" as const, content: "What is the weather in Boston?" }],
  tools: [
    {
      name: "get_weather",
      description: "Get the weather for a location",
      input_schema: {
        type: "object" as const,
        properties: { location: { type: "string" } },
        required: ["location"],
      },
    },
  ],
};

/**
 * Covers the dependent side of the GenAI semantic convention export: the
 * conversion lives in `@arizeai/openinference-core`, and these tests check
 * that it reaches the spans this instrumentation creates.
 */
describe("AnthropicInstrumentation with enableGenAISemconv", () => {
  let instrumentation: AnthropicInstrumentation | undefined;

  /**
   * Instruments the Anthropic SDK with the given trace config.
   * @param traceConfig - The trace config under test
   */
  function instrument(traceConfig?: TraceConfigOptions) {
    instrumentation = new AnthropicInstrumentation({ tracerProvider, traceConfig });
    instrumentation.disable();
    instrumentation._modules[0].moduleExports = Anthropic;
    instrumentation.enable();
  }

  /**
   * Waits for the only span of a test and reads its `gen_ai.*` attributes.
   * @returns The `gen_ai.*` attributes of the span
   */
  async function getExportedGenAIAttributes(): Promise<Record<string, unknown>> {
    await waitForSpansOn(memoryExporter, 1);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    return getGenAIAttributes(spans[0].attributes);
  }

  beforeEach(() => {
    memoryExporter.reset();
  });

  afterEach(() => {
    instrumentation?.disable();
  });

  it("should not emit gen_ai attributes by default", async () => {
    instrument();
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(toolUseMessage) });

    await client.messages.create(toolUseRequest);

    expect(await getExportedGenAIAttributes()).toEqual({});
  });

  it("should emit gen_ai attributes for a message with tool use", async () => {
    instrument({ enableGenAISemconv: true });
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(toolUseMessage) });

    await client.messages.create(toolUseRequest);

    expect(await getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "You are a weather assistant.",
                "type": "text",
              },
            ],
            "role": "system",
          },
          {
            "parts": [
              {
                "content": "What is the weather in Boston?",
                "type": "text",
              },
            ],
            "role": "user",
          },
        ],
        "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": [
          {
            "finish_reason": "tool_call",
            "parts": [
              {
                "content": "Let me check the weather.",
                "type": "text",
              },
              {
                "arguments": {
                  "location": "Boston",
                },
                "id": "toolu_weather",
                "name": "get_weather",
                "type": "tool_call",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.model": "claude-sonnet-4-6",
        "gen_ai.request.temperature": 0.2,
        "gen_ai.response.finish_reasons": [
          "tool_call",
        ],
        "gen_ai.response.id": "msg_tool_use",
        "gen_ai.response.model": "claude-sonnet-4-6",
        "gen_ai.tool.definitions": [
          {
            "description": "Get the weather for a location",
            "name": "get_weather",
            "parameters": {
              "properties": {
                "location": {
                  "type": "string",
                },
              },
              "required": [
                "location",
              ],
              "type": "object",
            },
            "type": "function",
          },
        ],
        "gen_ai.usage.cache_creation.input_tokens": 8,
        "gen_ai.usage.cache_read.input_tokens": 16,
        "gen_ai.usage.input_tokens": 64,
        "gen_ai.usage.output_tokens": 12,
      }
    `);
  });

  it("should emit gen_ai attributes for a streamed message", async () => {
    instrument({ enableGenAISemconv: true });
    const client = new Anthropic({
      apiKey: "fake-api-key",
      fetch: sseFetch(textStreamEvents()),
    });

    const stream = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      messages: [{ role: "user", content: "Say hello" }],
      stream: true,
    });
    for await (const _event of stream) {
      // Drain the stream so the span ends.
    }

    expect(await getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "Say hello",
                "type": "text",
              },
            ],
            "role": "user",
          },
        ],
        "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": [
          {
            "finish_reason": "stop",
            "parts": [
              {
                "content": "Hello, world",
                "type": "text",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.model": "claude-sonnet-4-6",
        "gen_ai.request.stream": true,
        "gen_ai.response.finish_reasons": [
          "stop",
        ],
        "gen_ai.response.id": "msg_stream",
        "gen_ai.response.model": "claude-sonnet-4-6",
        "gen_ai.usage.input_tokens": 10,
        "gen_ai.usage.output_tokens": 5,
      }
    `);
  });

  it("should derive gen_ai attributes from masked attributes", async () => {
    instrument({ enableGenAISemconv: true, hideInputs: true, hideOutputs: true });
    const client = new Anthropic({ apiKey: "fake-api-key", fetch: jsonFetch(toolUseMessage) });

    await client.messages.create(toolUseRequest);

    expect(await getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "anthropic",
        "gen_ai.request.max_tokens": 256,
        "gen_ai.request.model": "claude-sonnet-4-6",
        "gen_ai.request.temperature": 0.2,
        "gen_ai.response.finish_reasons": [
          "tool_call",
        ],
        "gen_ai.response.model": "claude-sonnet-4-6",
        "gen_ai.usage.cache_creation.input_tokens": 8,
        "gen_ai.usage.cache_read.input_tokens": 16,
        "gen_ai.usage.input_tokens": 64,
        "gen_ai.usage.output_tokens": 12,
      }
    `);
  });
});
