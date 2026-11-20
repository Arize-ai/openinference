import {
  ConverseCommand,
  ConverseStreamCommand,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import type { Attributes } from "@opentelemetry/api";
import { InMemorySpanExporter, SimpleSpanProcessor } from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import nock from "nock";

import type { TraceConfigOptions } from "@arizeai/openinference-core";

import { BedrockInstrumentation } from "../src/instrumentation";
import { TEST_MAX_TOKENS, TEST_MODEL_ID } from "./config/constants";
import { consumeStreamResponse } from "./helpers/test-helpers";
import { createTestClient, setupTestRecording } from "./helpers/vcr-helpers";

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

const weatherToolSchema = {
  type: "object" as const,
  properties: { location: { type: "string", description: "The city and state" } },
  required: ["location"],
};

/**
 * Covers the dependent side of the GenAI semantic convention export: the
 * conversion lives in `@arizeai/openinference-core`, and these tests check
 * that it reaches the spans this instrumentation creates. The requests replay
 * the recordings of `instrumentation.test.ts`.
 */
describe("BedrockInstrumentation with enableGenAISemconv", () => {
  const spanExporter = new InMemorySpanExporter();
  const provider = new NodeTracerProvider({
    spanProcessors: [new SimpleSpanProcessor(spanExporter)],
  });
  let instrumentation: BedrockInstrumentation | undefined;

  /**
   * Instruments the Bedrock runtime client with the given trace config.
   * @param traceConfig - The trace config under test
   */
  async function instrument(traceConfig?: TraceConfigOptions) {
    instrumentation = new BedrockInstrumentation({ traceConfig });
    instrumentation.disable();
    instrumentation.setTracerProvider(provider);
    const bedrockRuntime = await import("@aws-sdk/client-bedrock-runtime");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (instrumentation as any)._modules[0].moduleExports = bedrockRuntime;
    instrumentation.enable();
  }

  /**
   * Reads the `gen_ai.*` attributes of the only exported span.
   * @returns The `gen_ai.*` attributes of the span
   */
  function getExportedGenAIAttributes(): Record<string, unknown> {
    const spans = spanExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    return getGenAIAttributes(spans[0].attributes);
  }

  /**
   * Replays the recording of a test in `instrumentation.test.ts`.
   * @param recordingName - The name the recording was saved under
   */
  function replayRecording(recordingName: string) {
    setupTestRecording(recordingName, __dirname, false, TEST_MODEL_ID);
  }

  const invokeModelWithTools = () =>
    new InvokeModelCommand({
      modelId: TEST_MODEL_ID,
      body: JSON.stringify({
        anthropic_version: "bedrock-2023-05-31",
        max_tokens: TEST_MAX_TOKENS,
        tools: [
          {
            name: "get_weather",
            description: "Get current weather for a location",
            input_schema: weatherToolSchema,
          },
        ],
        messages: [{ role: "user", content: "What's the weather like in San Francisco?" }],
      }),
      contentType: "application/json",
      accept: "application/json",
    });

  beforeEach(() => {
    nock.cleanAll();
    if (!nock.isActive()) {
      nock.activate();
    }
    spanExporter.reset();
  });

  afterEach(() => {
    instrumentation?.disable();
    nock.cleanAll();
    nock.restore();
  });

  it("should not emit gen_ai attributes by default", async () => {
    await instrument();
    replayRecording("should handle tool calling with function definitions");

    await createTestClient(false).send(invokeModelWithTools());

    expect(getExportedGenAIAttributes()).toEqual({});
  });

  it("should emit gen_ai attributes for InvokeModel with tools", async () => {
    await instrument({ enableGenAISemconv: true });
    replayRecording("should handle tool calling with function definitions");

    await createTestClient(false).send(invokeModelWithTools());

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "What's the weather like in San Francisco?",
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
                "content": "Certainly! I can help you with that information. To get the current weather for San Francisco, I'll use the get_weather function. Let me fetch that data for you.",
                "type": "text",
              },
              {
                "arguments": {
                  "location": "San Francisco, CA",
                },
                "id": "toolu_bdrk_01MqHGzs8QwkdkVjJYrbLTPp",
                "name": "get_weather",
                "type": "tool_call",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "aws.bedrock",
        "gen_ai.request.max_tokens": 100,
        "gen_ai.request.model": "claude-3-5-sonnet-20240620",
        "gen_ai.response.finish_reasons": [
          "tool_call",
        ],
        "gen_ai.response.id": "msg_bdrk_01UCkMJ9Yp7J1ZpAWtK8CdYN",
        "gen_ai.response.model": "claude-3-5-sonnet-20240620",
        "gen_ai.tool.definitions": [
          {
            "description": "Get current weather for a location",
            "name": "get_weather",
            "parameters": {
              "properties": {
                "location": {
                  "description": "The city and state",
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
        "gen_ai.usage.input_tokens": 373,
        "gen_ai.usage.output_tokens": 94,
      }
    `);
  });

  it("should emit gen_ai attributes for InvokeModelWithResponseStream", async () => {
    await instrument({ enableGenAISemconv: true });
    replayRecording("should handle invoke model with response stream");

    const result = await createTestClient(false).send(
      new InvokeModelWithResponseStreamCommand({
        modelId: TEST_MODEL_ID,
        body: JSON.stringify({
          anthropic_version: "bedrock-2023-05-31",
          max_tokens: 100,
          messages: [{ role: "user", content: "Tell me a short story" }],
        }),
        contentType: "application/json",
        accept: "application/json",
      }),
    );
    await consumeStreamResponse(result);

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "Tell me a short story",
                "type": "text",
              },
            ],
            "role": "user",
          },
        ],
        "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": [
          {
            "finish_reason": "length",
            "parts": [
              {
                "content": "Here's a short story for you:

      The Last Leaf

      Ella gazed out her window at the ivy-covered wall across the courtyard. She had been bedridden with pneumonia for weeks, and her spirits were low. The doctor had told her that she needed the will to live to recover, but Ella felt herself slipping away.

      She had been counting the ivy leaves as they fell, convinced that when the last leaf dropped, she too would die. Now",
                "type": "text",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "aws.bedrock",
        "gen_ai.request.max_tokens": 100,
        "gen_ai.request.model": "claude-3-5-sonnet-20240620",
        "gen_ai.response.finish_reasons": [
          "length",
        ],
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 100,
      }
    `);
  });

  it("should emit gen_ai attributes for Converse with an inference config", async () => {
    await instrument({ enableGenAISemconv: true });
    replayRecording("should-handle-inference-config-in-converse-api");

    await createTestClient(false).send(
      new ConverseCommand({
        modelId: TEST_MODEL_ID,
        messages: [{ role: "user", content: [{ text: "Explain machine learning briefly." }] }],
        inferenceConfig: {
          maxTokens: 150,
          temperature: 0.7,
          topP: 0.9,
          stopSequences: ["END", "STOP"],
        },
      }),
    );

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "Explain machine learning briefly.",
                "type": "text",
              },
            ],
            "role": "user",
          },
        ],
        "gen_ai.operation.name": "chat",
        "gen_ai.output.messages": [
          {
            "finish_reason": "length",
            "parts": [
              {
                "content": "Machine learning is a branch of artificial intelligence that focuses on developing algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience, without being explicitly programmed.

      In essence, machine learning allows computers to learn from data and make predictions or decisions without human intervention. Here's a brief overview of the key aspects of machine learning:

      1. Types of Machine Learning:
         - Supervised Learning: The algorithm learns from labeled data to make predictions or classifications.
         - Unsupervised Learning: The algorithm finds patterns in unlabeled data.
         - Reinforcement Learning: The algorithm learns through interaction with an environment, receiving feedback in the form of rewards or penalties.

      2. Process:",
                "type": "text",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "aws.bedrock",
        "gen_ai.request.max_tokens": 150,
        "gen_ai.request.model": "claude-3-5-sonnet-20240620",
        "gen_ai.request.stop_sequences": [
          "END",
          "STOP",
        ],
        "gen_ai.request.temperature": 0.7,
        "gen_ai.request.top_p": 0.9,
        "gen_ai.response.finish_reasons": [
          "length",
        ],
        "gen_ai.usage.input_tokens": 13,
        "gen_ai.usage.output_tokens": 150,
      }
    `);
  });

  it("should emit gen_ai attributes for ConverseStream with tools", async () => {
    await instrument({ enableGenAISemconv: true });
    replayRecording("should-handle-streaming-tool-calls-and-responses");

    const result = await createTestClient(false).send(
      new ConverseStreamCommand({
        modelId: TEST_MODEL_ID,
        messages: [
          {
            role: "user",
            content: [{ text: "What's the weather in San Francisco and what time is it there?" }],
          },
        ],
        toolConfig: {
          tools: [
            {
              toolSpec: {
                name: "get_weather",
                description: "Get current weather for a location",
                inputSchema: { json: weatherToolSchema },
              },
            },
          ],
        },
      }),
    );
    await consumeStreamResponse({ body: result.stream });

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.input.messages": [
          {
            "parts": [
              {
                "content": "What's the weather in San Francisco and what time is it there?",
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
                "content": "I can certainly help you with the weather in San Francisco, but I'm afraid I don't have a specific tool to check the current time there. Let me get the weather information for you.",
                "type": "text",
              },
              {
                "arguments": {
                  "location": "San Francisco, CA",
                  "unit": "fahrenheit",
                },
                "id": "tooluse_wnjikhCUTruJciycmmm5Kg",
                "name": "get_weather",
                "type": "tool_call",
              },
            ],
            "role": "assistant",
          },
        ],
        "gen_ai.provider.name": "aws.bedrock",
        "gen_ai.request.model": "claude-3-5-sonnet-20240620",
        "gen_ai.response.finish_reasons": [
          "tool_call",
        ],
        "gen_ai.tool.definitions": [
          {
            "description": "Get current weather for a location",
            "name": "get_weather",
            "parameters": {
              "properties": {
                "location": {
                  "description": "The city and state",
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
        "gen_ai.usage.input_tokens": 414,
        "gen_ai.usage.output_tokens": 114,
      }
    `);
  });

  it("should derive gen_ai attributes from masked attributes", async () => {
    await instrument({ enableGenAISemconv: true, hideInputs: true, hideOutputs: true });
    replayRecording("should handle tool calling with function definitions");

    await createTestClient(false).send(invokeModelWithTools());

    expect(getExportedGenAIAttributes()).toMatchInlineSnapshot(`
      {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "aws.bedrock",
        "gen_ai.request.max_tokens": 100,
        "gen_ai.request.model": "claude-3-5-sonnet-20240620",
        "gen_ai.response.finish_reasons": [
          "tool_call",
        ],
        "gen_ai.usage.input_tokens": 373,
        "gen_ai.usage.output_tokens": 94,
      }
    `);
  });
});
