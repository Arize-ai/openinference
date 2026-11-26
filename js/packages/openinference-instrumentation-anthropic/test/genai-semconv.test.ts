import Anthropic from "@anthropic-ai/sdk";
import type { Attributes } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { TraceConfigOptions } from "@arizeai/openinference-core";

import { AnthropicInstrumentation } from "../src/instrumentation";
import { getAPIKey, resetRecordings, startRecording } from "./helpers/recordings";
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

/** A request the model answers with a call to `get_weather`. */
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
 * A system prompt long enough to be eligible for prompt caching, which
 * Anthropic only applies above a model specific minimum prompt length.
 */
const cacheableInstructions = Array.from(
  { length: 300 },
  (_, index) =>
    `Guideline ${index + 1}: when asked about the weather in region ${index + 1}, answer in one short sentence.`,
).join("\n");

/** The usage attributes a prompt caching snapshot keeps. */
const USAGE_ATTRIBUTES = [
  "gen_ai.usage.cache_creation.input_tokens",
  "gen_ai.usage.cache_read.input_tokens",
  "gen_ai.usage.input_tokens",
  "gen_ai.usage.output_tokens",
];

/**
 * Picks the token usage attributes out of the `gen_ai.*` attributes of a span.
 * @param genAIAttributes - The `gen_ai.*` attributes of a span
 * @returns The usage attributes that are set, keyed by attribute name
 */
function getUsageAttributes(genAIAttributes: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(genAIAttributes).filter(([key]) => USAGE_ATTRIBUTES.includes(key)),
  );
}

/**
 * Covers the dependent side of the GenAI semantic convention export: the
 * conversion lives in `@arizeai/openinference-core`, and these tests check
 * that it reaches the spans this instrumentation creates. The requests replay
 * the network recordings in `test/recordings`; see `helpers/recordings.ts` for
 * how to capture them again.
 */
describe("AnthropicInstrumentation with enableGenAISemconv", () => {
  let instrumentation: AnthropicInstrumentation | undefined;
  let client: Anthropic;

  /**
   * Instruments the Anthropic SDK with the given trace config.
   * @param traceConfig - The trace config under test
   */
  function instrument(traceConfig?: TraceConfigOptions) {
    instrumentation = new AnthropicInstrumentation({ tracerProvider, traceConfig });
    instrumentation.disable();
    instrumentation._modules[0].moduleExports = Anthropic;
    instrumentation.enable();
    // Retries are off so a request without a recording fails at once.
    client = new Anthropic({ apiKey: getAPIKey("ANTHROPIC_API_KEY"), maxRetries: 0 });
  }

  /**
   * Waits for the spans of a test and reads their `gen_ai.*` attributes.
   * @param expectedSpanCount - The number of spans the test should export
   * @returns The `gen_ai.*` attributes of each span, in export order
   */
  async function getExportedGenAIAttributesBySpan(
    expectedSpanCount: number,
  ): Promise<Record<string, unknown>[]> {
    await waitForSpansOn(memoryExporter, expectedSpanCount);
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(expectedSpanCount);
    return spans.map((span) => getGenAIAttributes(span.attributes));
  }

  /**
   * Waits for the only span of a test and reads its `gen_ai.*` attributes.
   * @returns The `gen_ai.*` attributes of the span
   */
  async function getExportedGenAIAttributes(): Promise<Record<string, unknown>> {
    const [genAIAttributes] = await getExportedGenAIAttributesBySpan(1);
    return genAIAttributes;
  }

  beforeEach(() => {
    memoryExporter.reset();
  });

  afterEach(() => {
    instrumentation?.disable();
    resetRecordings();
  });

  it("should not emit gen_ai attributes by default", async () => {
    instrument();
    const recording = await startRecording("genai-message-tool-use");

    await client.messages.create(toolUseRequest);
    recording.stop();

    expect(await getExportedGenAIAttributes()).toEqual({});
  });

  it("should emit gen_ai attributes for a message with tool use", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-message-tool-use");

    await client.messages.create(toolUseRequest);
    recording.stop();

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
                "arguments": {
                  "location": "Boston",
                },
                "id": "toolu_01CyRuseU7wBCkorwdreggF7",
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
        "gen_ai.response.id": "msg_011ChErfnseoSPfvAE3VBwBE",
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
        "gen_ai.usage.input_tokens": 572,
        "gen_ai.usage.output_tokens": 54,
      }
    `);
  });

  it("should emit gen_ai attributes for a streamed message", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-message-stream");

    const stream = await client.messages.create({
      model: "claude-sonnet-4-6",
      max_tokens: 256,
      messages: [{ role: "user", content: "Say hello" }],
      stream: true,
    });
    for await (const _event of stream) {
      // Drain the stream so the span ends.
    }
    recording.stop();

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
                "content": "Hello! 👋 How are you doing? Is there something I can help you with today?",
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
        "gen_ai.response.id": "msg_011ChErftB8uNkKjCb7T1Z1k",
        "gen_ai.response.model": "claude-sonnet-4-6",
        "gen_ai.usage.input_tokens": 9,
        "gen_ai.usage.output_tokens": 24,
      }
    `);
  });

  it("should emit cache creation and cache read tokens for a cached prompt", async () => {
    instrument({ enableGenAISemconv: true });
    const recording = await startRecording("genai-message-cached-prompt");
    const cachedPromptRequest = {
      model: "claude-sonnet-4-6",
      max_tokens: 32,
      system: [
        {
          type: "text" as const,
          text: cacheableInstructions,
          cache_control: { type: "ephemeral" as const },
        },
      ],
      messages: [{ role: "user" as const, content: "What is the weather in region 7?" }],
    };

    // The first request writes the prompt cache and the second reads it.
    await client.messages.create(cachedPromptRequest);
    await client.messages.create(cachedPromptRequest);
    recording.stop();

    const genAIAttributesBySpan = await getExportedGenAIAttributesBySpan(2);
    expect(genAIAttributesBySpan.map(getUsageAttributes)).toMatchInlineSnapshot(`
      [
        {
          "gen_ai.usage.cache_creation.input_tokens": 6901,
          "gen_ai.usage.input_tokens": 6916,
          "gen_ai.usage.output_tokens": 17,
        },
        {
          "gen_ai.usage.cache_read.input_tokens": 6901,
          "gen_ai.usage.input_tokens": 6916,
          "gen_ai.usage.output_tokens": 17,
        },
      ]
    `);
  });

  it("should derive gen_ai attributes from masked attributes", async () => {
    instrument({ enableGenAISemconv: true, hideInputs: true, hideOutputs: true });
    const recording = await startRecording("genai-message-tool-use");

    await client.messages.create(toolUseRequest);
    recording.stop();

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
        "gen_ai.usage.input_tokens": 572,
        "gen_ai.usage.output_tokens": 54,
      }
    `);
  });
});
