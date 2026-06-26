import type { Attributes } from "@opentelemetry/api";
import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import { BasicTracerProvider, InMemorySpanExporter } from "@opentelemetry/sdk-trace-base";
import { afterEach, beforeEach, describe, expect, it, test } from "vitest";

import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import type { SpanFilter } from "../src";
import {
  isOpenInferenceSpan,
  OpenInferenceBatchSpanProcessor,
  OpenInferenceSimpleSpanProcessor,
} from "../src";
import { VercelSDKFunctionNameToSpanKindMap } from "../src/constants";
import { VercelAISemanticConventions } from "../src/VercelAISemanticConventions";
import embedDoEmbedFixture from "./__fixtures__/v6-spans/ai-embed-doEmbed.json";
import generateObjectDoGenerateFixture from "./__fixtures__/v6-spans/ai-generateObject-doGenerate.json";
// Import real AI SDK v6 fixtures
import generateTextDoGenerateFixture from "./__fixtures__/v6-spans/ai-generateText-doGenerate.json";
import streamTextDoStreamFixture from "./__fixtures__/v6-spans/ai-streamText-doStream.json";

type SpanProcessorTestCase = [
  string,
  {
    vercelFunctionName: string;
    vercelAttributes: Attributes;
    expectedOpenInferenceAttributes: Partial<Attributes>;
  },
];

/**
 * Generate test cases from real AI SDK v6 fixtures
 */
const generateV6FixtureTestCases = (): SpanProcessorTestCase[] => {
  const testCases: SpanProcessorTestCase[] = [];

  // Test generateText.doGenerate
  const generateTextSpan = generateTextDoGenerateFixture[0];
  testCases.push([
    "AI SDK v6 generateText.doGenerate span",
    {
      vercelFunctionName: "ai.generateText.doGenerate",
      vercelAttributes: generateTextSpan.attributes as Attributes,
      expectedOpenInferenceAttributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        // gen_ai.response.model takes precedence over gen_ai.request.model
        [SemanticConventions.LLM_MODEL_NAME]: "gpt-4o-mini-2024-07-18",
        // gen_ai.usage.* should be converted
        [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 14,
        [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 20,
        // Metadata should be extracted
        [`${SemanticConventions.METADATA}.testCategory`]: "text-generation",
        [`${SemanticConventions.METADATA}.customField`]: "custom-value",
        // Output should be set
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
      },
    },
  ]);

  // Test streamText.doStream
  const streamTextSpan = streamTextDoStreamFixture[0];
  testCases.push([
    "AI SDK v6 streamText.doStream span with streaming metrics",
    {
      vercelFunctionName: "ai.streamText.doStream",
      vercelAttributes: streamTextSpan.attributes as Attributes,
      expectedOpenInferenceAttributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        // gen_ai.response.model takes precedence over gen_ai.request.model
        [SemanticConventions.LLM_MODEL_NAME]: "gpt-4o-mini-2024-07-18",
        // Streaming metrics should be stored as metadata
        [`${SemanticConventions.METADATA}.ai.response.msToFirstChunk`]:
          streamTextSpan.attributes["ai.response.msToFirstChunk"],
        [`${SemanticConventions.METADATA}.ai.response.msToFinish`]:
          streamTextSpan.attributes["ai.response.msToFinish"],
      },
    },
  ]);

  // Test embed.doEmbed
  const embedSpan = embedDoEmbedFixture[0];
  testCases.push([
    "AI SDK v6 embed.doEmbed span",
    {
      vercelFunctionName: "ai.embed.doEmbed",
      vercelAttributes: embedSpan.attributes as Attributes,
      expectedOpenInferenceAttributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.EMBEDDING,
        [SemanticConventions.EMBEDDING_MODEL_NAME]: "text-embedding-3-small",
      },
    },
  ]);

  // Test generateObject.doGenerate
  const generateObjectSpan = generateObjectDoGenerateFixture[0];
  testCases.push([
    "AI SDK v6 generateObject.doGenerate span",
    {
      vercelFunctionName: "ai.generateObject.doGenerate",
      vercelAttributes: generateObjectSpan.attributes as Attributes,
      expectedOpenInferenceAttributes: {
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
      },
    },
  ]);

  return testCases;
};

/**
 * Generate test cases for Vercel-specific attribute handling
 */
const generateVercelAttributeTestCases = (): SpanProcessorTestCase[] => {
  const testCases: SpanProcessorTestCase[] = [];

  // Model ID mapping
  testCases.push(
    [
      `${VercelAISemanticConventions.MODEL_ID} to ${SemanticConventions.LLM_MODEL_NAME} for LLM`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.MODEL_ID]: "test-llm",
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.LLM_MODEL_NAME]: "test-llm",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.MODEL_ID} to ${SemanticConventions.EMBEDDING_MODEL_NAME} for embeddings`,
      {
        vercelFunctionName: "ai.embed.doEmbed",
        vercelAttributes: {
          [VercelAISemanticConventions.MODEL_ID]: "test-embedding-model",
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.EMBEDDING_MODEL_NAME]: "test-embedding-model",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.EMBEDDING,
        },
      },
    ],
  );

  // Metadata mapping
  testCases.push([
    `${VercelAISemanticConventions.METADATA} to ${SemanticConventions.METADATA}`,
    {
      vercelFunctionName: "ai.generateText.doGenerate",
      vercelAttributes: {
        [`${VercelAISemanticConventions.METADATA}.key1`]: "value1",
        [`${VercelAISemanticConventions.METADATA}.key2`]: "value2",
      },
      expectedOpenInferenceAttributes: {
        [`${SemanticConventions.METADATA}.key1`]: "value1",
        [`${SemanticConventions.METADATA}.key2`]: "value2",
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      },
    },
  ]);

  // Settings to invocation parameters
  testCases.push([
    `${VercelAISemanticConventions.SETTINGS} to ${SemanticConventions.LLM_INVOCATION_PARAMETERS}`,
    {
      vercelFunctionName: "ai.generateText.doGenerate",
      vercelAttributes: {
        [`${VercelAISemanticConventions.SETTINGS}.key1`]: "value1",
        [`${VercelAISemanticConventions.SETTINGS}.key2`]: "value2",
      },
      expectedOpenInferenceAttributes: {
        [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify({
          key1: "value1",
          key2: "value2",
        }),
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      },
    },
  ]);

  // Token counts
  testCases.push(
    [
      `${VercelAISemanticConventions.TOKEN_COUNT_COMPLETION} to ${SemanticConventions.LLM_TOKEN_COUNT_COMPLETION}`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.TOKEN_COUNT_COMPLETION]: 10,
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 10,
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.TOKEN_COUNT_PROMPT} to ${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.TOKEN_COUNT_PROMPT]: 10,
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 10,
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    [
      "Token counts should not be mapped for CHAIN spans",
      {
        vercelFunctionName: "ai.embed",
        vercelAttributes: {
          [VercelAISemanticConventions.TOKEN_COUNT_COMPLETION]: 10,
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.CHAIN,
        },
      },
    ],
  );

  // Response text to output
  testCases.push([
    `${VercelAISemanticConventions.RESPONSE_TEXT} to ${SemanticConventions.OUTPUT_VALUE}`,
    {
      vercelFunctionName: "ai.generateText.doGenerate",
      vercelAttributes: {
        [VercelAISemanticConventions.RESPONSE_TEXT]: "hello",
      },
      expectedOpenInferenceAttributes: {
        [SemanticConventions.OUTPUT_VALUE]: "hello",
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      },
    },
  ]);

  // Response object to output
  testCases.push([
    `${VercelAISemanticConventions.RESPONSE_OBJECT} to ${SemanticConventions.OUTPUT_VALUE}`,
    {
      vercelFunctionName: "ai.generateObject.doGenerate",
      vercelAttributes: {
        [VercelAISemanticConventions.RESPONSE_OBJECT]: JSON.stringify({
          key: "value",
        }),
      },
      expectedOpenInferenceAttributes: {
        [SemanticConventions.OUTPUT_VALUE]: JSON.stringify({ key: "value" }),
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      },
    },
  ]);

  // Prompt to input
  testCases.push(
    [
      `${VercelAISemanticConventions.PROMPT} to ${SemanticConventions.INPUT_VALUE} (text)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT]: "hello",
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.INPUT_VALUE]: "hello",
          [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.PROMPT} to ${SemanticConventions.INPUT_VALUE} (JSON)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT]: JSON.stringify({}),
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.INPUT_VALUE]: "{}",
          [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
  );

  // Prompt messages to input messages
  const firstInputMessageContentsPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}`;
  testCases.push(
    [
      `${VercelAISemanticConventions.PROMPT_MESSAGES} to ${SemanticConventions.LLM_INPUT_MESSAGES} (string content)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT_MESSAGES]: JSON.stringify([
            { role: "assistant", content: "hello" },
            { role: "user", content: "world" },
          ]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "assistant",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            "hello",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]:
            "user",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
            "world",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.PROMPT_MESSAGES} to ${SemanticConventions.LLM_INPUT_MESSAGES} (object content)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT_MESSAGES]: JSON.stringify([
            {
              role: "assistant",
              content: [
                { type: "text", text: "hello" },
                { type: "image", image: "image.com" },
              ],
            },
          ]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "assistant",
          [`${firstInputMessageContentsPrefix}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
            "text",
          [`${firstInputMessageContentsPrefix}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`]:
            "hello",
          [`${firstInputMessageContentsPrefix}.1.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
            "image",
          [`${firstInputMessageContentsPrefix}.1.${SemanticConventions.MESSAGE_CONTENT_IMAGE}`]:
            "image.com",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.PROMPT_MESSAGES} to ${SemanticConventions.LLM_INPUT_MESSAGES} (tool calls)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT_MESSAGES]: JSON.stringify([
            {
              role: "assistant",
              content: [
                {
                  type: "tool_call",
                  toolCallId: "test-tool-id",
                  toolName: "test-tool",
                  input: { testInput: "test" },
                },
              ],
            },
          ]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "assistant",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
            JSON.stringify({ testInput: "test" }),
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
            "test-tool",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_ID}`]:
            "test-tool-id",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    // Tool result messages (role: "tool") - AI SDK v6 format with output
    [
      `${VercelAISemanticConventions.PROMPT_MESSAGES} to ${SemanticConventions.LLM_INPUT_MESSAGES} (tool results with output)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT_MESSAGES]: JSON.stringify([
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: "test-tool-call-id",
                  toolName: "weather",
                  output: { location: "Boston", temperature: 72 },
                },
              ],
            },
          ]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "tool",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]:
            "test-tool-call-id",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            JSON.stringify({ location: "Boston", temperature: 72 }),
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    // Tool result messages with string output
    [
      `${VercelAISemanticConventions.PROMPT_MESSAGES} to ${SemanticConventions.LLM_INPUT_MESSAGES} (tool results with string output)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT_MESSAGES]: JSON.stringify([
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: "calc-tool-id",
                  toolName: "calculator",
                  output: "2503",
                },
              ],
            },
          ]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "tool",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]:
            "calc-tool-id",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            "2503",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    // Legacy tool result format with 'result' property
    [
      `${VercelAISemanticConventions.PROMPT_MESSAGES} to ${SemanticConventions.LLM_INPUT_MESSAGES} (tool results with legacy result property)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT_MESSAGES]: JSON.stringify([
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: "legacy-tool-id",
                  toolName: "legacyTool",
                  result: { data: "legacy result" },
                },
              ],
            },
          ]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "tool",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]:
            "legacy-tool-id",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            JSON.stringify({ data: "legacy result" }),
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    // Multiple tool result messages (separate messages for each tool result)
    [
      `${VercelAISemanticConventions.PROMPT_MESSAGES} to ${SemanticConventions.LLM_INPUT_MESSAGES} (multiple tool result messages)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT_MESSAGES]: JSON.stringify([
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: "weather-call-id",
                  toolName: "weather",
                  output: { location: "Boston", temperature: 72 },
                },
              ],
            },
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: "calculator-call-id",
                  toolName: "calculator",
                  output: { expression: "100 * 25 + 3", value: 2503 },
                },
              ],
            },
          ]),
        },
        expectedOpenInferenceAttributes: {
          // First tool result message
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "tool",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]:
            "weather-call-id",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            JSON.stringify({ location: "Boston", temperature: 72 }),
          // Second tool result message
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]:
            "tool",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]:
            "calculator-call-id",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
            JSON.stringify({ expression: "100 * 25 + 3", value: 2503 }),
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
    // Multiple tool results within a single Vercel message get expanded into separate OpenInference messages
    // Per OpenInference spec, each tool result should be a separate message with:
    // - message.role: "tool"
    // - message.content: the result content
    // - message.tool_call_id: linking back to the original tool call
    [
      `${VercelAISemanticConventions.PROMPT_MESSAGES} to ${SemanticConventions.LLM_INPUT_MESSAGES} (multiple tool results expanded to separate messages)`,
      {
        vercelFunctionName: "ai.generateText.doGenerate",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT_MESSAGES]: JSON.stringify([
            {
              role: "tool",
              content: [
                {
                  type: "tool-result",
                  toolCallId: "weather-call-id",
                  toolName: "weather",
                  output: { location: "Boston", temperature: 72 },
                },
                {
                  type: "tool-result",
                  toolCallId: "calculator-call-id",
                  toolName: "calculator",
                  output: { expression: "100 * 25 + 3", value: 2503 },
                },
              ],
            },
          ]),
        },
        expectedOpenInferenceAttributes: {
          // First tool result becomes message index 0
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "tool",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            JSON.stringify({ location: "Boston", temperature: 72 }),
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]:
            "weather-call-id",
          // Second tool result becomes message index 1
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]:
            "tool",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
            JSON.stringify({ expression: "100 * 25 + 3", value: 2503 }),
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_TOOL_CALL_ID}`]:
            "calculator-call-id",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
        },
      },
    ],
  );

  // Response tool calls to output messages
  const firstOutputMessageToolPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}`;
  testCases.push([
    `${VercelAISemanticConventions.RESPONSE_TOOL_CALLS} to ${SemanticConventions.LLM_OUTPUT_MESSAGES} tool calls`,
    {
      vercelFunctionName: "ai.toolCall",
      vercelAttributes: {
        [VercelAISemanticConventions.RESPONSE_TOOL_CALLS]: JSON.stringify([
          {
            toolCallId: "call_1",
            toolName: "test-tool-1",
            args: { test1: "test-1" },
          },
          {
            toolCallId: "call_2",
            toolName: "test-tool-2",
            input: { test2: "test-2" },
          },
        ]),
      },
      expectedOpenInferenceAttributes: {
        [`${firstOutputMessageToolPrefix}.0.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
          "test-tool-1",
        [`${firstOutputMessageToolPrefix}.0.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
          JSON.stringify({ test1: "test-1" }),
        [`${firstOutputMessageToolPrefix}.0.${SemanticConventions.TOOL_CALL_ID}`]: "call_1",
        [`${firstOutputMessageToolPrefix}.1.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
          "test-tool-2",
        [`${firstOutputMessageToolPrefix}.1.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
          JSON.stringify({ test2: "test-2" }),
        [`${firstOutputMessageToolPrefix}.1.${SemanticConventions.TOOL_CALL_ID}`]: "call_2",
        [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
          "assistant",
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
      },
    },
  ]);

  // Input messages extracted from ai.prompt (outer AGENT spans like ai.streamText, ai.generateText)
  testCases.push(
    [
      `${VercelAISemanticConventions.PROMPT} to ${SemanticConventions.LLM_INPUT_MESSAGES} (messages with system)`,
      {
        vercelFunctionName: "ai.streamText",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT]: JSON.stringify({
            system: "You are a helpful assistant.",
            messages: [{ role: "user", content: "What is the weather?" }],
          }),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "system",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            "You are a helpful assistant.",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]:
            "user",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
            "What is the weather?",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.PROMPT} to ${SemanticConventions.LLM_INPUT_MESSAGES} (messages array only)`,
      {
        vercelFunctionName: "ai.generateText",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT]: JSON.stringify([
            { role: "user", content: "Hello" },
          ]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "user",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            "Hello",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.PROMPT} to ${SemanticConventions.LLM_INPUT_MESSAGES} (messages without system)`,
      {
        vercelFunctionName: "ai.streamText",
        vercelAttributes: {
          [VercelAISemanticConventions.PROMPT]: JSON.stringify({
            messages: [
              { role: "user", content: "Tell me a joke" },
              {
                role: "assistant",
                content: "Why did the chicken cross the road?",
              },
            ],
          }),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
            "user",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
            "Tell me a joke",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]:
            "assistant",
          [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
            "Why did the chicken cross the road?",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
        },
      },
    ],
  );

  // Output value fallback from tool calls when response text is empty
  testCases.push([
    `${VercelAISemanticConventions.RESPONSE_TOOL_CALLS} falls back to ${SemanticConventions.OUTPUT_VALUE} when response text is empty`,
    {
      vercelFunctionName: "ai.streamText.doStream",
      vercelAttributes: {
        [VercelAISemanticConventions.RESPONSE_TEXT]: "",
        [VercelAISemanticConventions.RESPONSE_TOOL_CALLS]: JSON.stringify([
          {
            toolCallId: "call_abc",
            toolName: "get_weather",
            args: { location: "Boston" },
          },
        ]),
      },
      expectedOpenInferenceAttributes: {
        [SemanticConventions.OUTPUT_VALUE]: JSON.stringify([
          {
            toolCallId: "call_abc",
            toolName: "get_weather",
            args: { location: "Boston" },
          },
        ]),
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      },
    },
  ]);

  // Input value fallback from prompt messages on LLM spans
  const promptMessagesJSON = JSON.stringify([
    { role: "system", content: "You are helpful." },
    { role: "user", content: "Hi" },
  ]);
  testCases.push([
    `${VercelAISemanticConventions.PROMPT_MESSAGES} falls back to ${SemanticConventions.INPUT_VALUE} when ai.prompt is absent`,
    {
      vercelFunctionName: "ai.generateText.doGenerate",
      vercelAttributes: {
        [VercelAISemanticConventions.PROMPT_MESSAGES]: promptMessagesJSON,
      },
      expectedOpenInferenceAttributes: {
        [SemanticConventions.INPUT_VALUE]: promptMessagesJSON,
        [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
          "system",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
          "You are helpful.",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]: "user",
        [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
          "Hi",
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
      },
    },
  ]);

  // Embeddings
  testCases.push(
    [
      `${VercelAISemanticConventions.EMBEDDING_TEXT} to ${SemanticConventions.EMBEDDING_TEXT}`,
      {
        vercelFunctionName: "ai.embed.doEmbed",
        vercelAttributes: {
          [VercelAISemanticConventions.EMBEDDING_TEXT]: "hello",
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
            "hello",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.EMBEDDING,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.EMBEDDING_TEXTS} to ${SemanticConventions.EMBEDDING_TEXT} (multiple)`,
      {
        vercelFunctionName: "ai.embedMany.doEmbed",
        vercelAttributes: {
          [VercelAISemanticConventions.EMBEDDING_TEXTS]: ["hello", "world"],
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
            "hello",
          [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_TEXT}`]:
            "world",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.EMBEDDING,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.EMBEDDING_VECTOR} to ${SemanticConventions.EMBEDDING_VECTOR}`,
      {
        vercelFunctionName: "ai.embedMany.doEmbed",
        vercelAttributes: {
          [VercelAISemanticConventions.EMBEDDING_VECTOR]: JSON.stringify([1, 2]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
            [1, 2],
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.EMBEDDING,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.EMBEDDING_VECTORS} to ${SemanticConventions.EMBEDDING_VECTOR} (multiple)`,
      {
        vercelFunctionName: "ai.embedMany.doEmbed",
        vercelAttributes: {
          [VercelAISemanticConventions.EMBEDDING_VECTORS]: ["[1, 2]", "[3, 4]"],
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
            [1, 2],
          [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_VECTOR}`]:
            [3, 4],
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.EMBEDDING,
        },
      },
    ],
  );

  // Tool call spans
  testCases.push(
    [
      `${VercelAISemanticConventions.TOOL_CALL_ID} to ${SemanticConventions.TOOL_CALL_ID}`,
      {
        vercelFunctionName: "ai.toolCall",
        vercelAttributes: {
          [VercelAISemanticConventions.TOOL_CALL_ID]: "test-tool-id",
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.TOOL_CALL_ID]: "test-tool-id",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.TOOL_CALL_NAME} to ${SemanticConventions.TOOL_NAME}`,
      {
        vercelFunctionName: "ai.toolCall",
        vercelAttributes: {
          [VercelAISemanticConventions.TOOL_CALL_NAME]: "test-tool",
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.TOOL_NAME]: "test-tool",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.TOOL_CALL_ARGS} to ${SemanticConventions.TOOL_PARAMETERS} and ${SemanticConventions.INPUT_VALUE}`,
      {
        vercelFunctionName: "ai.toolCall",
        vercelAttributes: {
          [VercelAISemanticConventions.TOOL_CALL_ARGS]: JSON.stringify({
            test1: "test-1",
          }),
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify({
            test1: "test-1",
          }),
          [SemanticConventions.INPUT_VALUE]: JSON.stringify({
            test1: "test-1",
          }),
          [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.TOOL_CALL_RESULT} to ${SemanticConventions.OUTPUT_VALUE}`,
      {
        vercelFunctionName: "ai.toolCall",
        vercelAttributes: {
          [VercelAISemanticConventions.TOOL_CALL_RESULT]: "test-result",
        },
        expectedOpenInferenceAttributes: {
          [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
          [SemanticConventions.OUTPUT_VALUE]: "test-result",
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
        },
      },
    ],
  );

  return testCases;
};

let traceProvider = new BasicTracerProvider();
let memoryExporter = new InMemorySpanExporter();
let processor: OpenInferenceSimpleSpanProcessor | OpenInferenceBatchSpanProcessor;

function setupTraceProvider({
  Processor,
  spanFilter,
}: {
  Processor: typeof OpenInferenceBatchSpanProcessor | typeof OpenInferenceSimpleSpanProcessor;
  spanFilter?: SpanFilter;
}) {
  memoryExporter.reset();
  trace.disable();
  memoryExporter = new InMemorySpanExporter();
  processor = new Processor({
    exporter: memoryExporter,
    spanFilter,
  });
  traceProvider = new BasicTracerProvider({ spanProcessors: [processor] });
  trace.setGlobalTracerProvider(traceProvider);
}

describe("OpenInferenceSimpleSpanProcessor", () => {
  beforeEach(() => {
    setupTraceProvider({ Processor: OpenInferenceSimpleSpanProcessor });
  });
  afterEach(() => {
    trace.disable();
  });

  it("should get the span kind from attributes", () => {
    const tracer = trace.getTracer("test-tracer");
    VercelSDKFunctionNameToSpanKindMap.forEach((spanKind, functionName) => {
      const span = tracer.startSpan(functionName);
      span.setAttribute("operation.name", functionName);
      span.end();
      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(spanKind);
      memoryExporter.reset();
    });
  });

  test.each(generateVercelAttributeTestCases())(
    "should map %s",
    (_name, { vercelFunctionName, vercelAttributes, expectedOpenInferenceAttributes }) => {
      const tracer = trace.getTracer("test-tracer");
      const span = tracer.startSpan(vercelFunctionName);
      const vercelAttributesWithOperationName = {
        ...vercelAttributes,
        "operation.name": vercelFunctionName,
      };
      span.setAttributes(vercelAttributesWithOperationName);
      span.end();
      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      // Check that expected attributes are present
      Object.entries(expectedOpenInferenceAttributes).forEach(([key, value]) => {
        expect(spans[0].attributes[key]).toEqual(value);
      });
    },
  );

  test.each(generateV6FixtureTestCases())(
    "should correctly process %s",
    (_name, { vercelFunctionName, vercelAttributes, expectedOpenInferenceAttributes }) => {
      const tracer = trace.getTracer("test-tracer");
      const span = tracer.startSpan(vercelFunctionName);
      span.setAttributes(vercelAttributes);
      span.end();
      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      // Check that expected attributes are present
      Object.entries(expectedOpenInferenceAttributes).forEach(([key, value]) => {
        expect(spans[0].attributes[key]).toEqual(value);
      });
    },
  );

  it("should not export non-AI spans", () => {
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("ai.generateText");
    span.setAttribute("operation.name", "ai.generateText");
    span.end();
    const nonOpenInferenceSpan = tracer.startSpan("non-ai-span");
    nonOpenInferenceSpan.end();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(2);
  });

  it("should export all spans if there is no filter", () => {
    setupTraceProvider({
      Processor: OpenInferenceSimpleSpanProcessor,
    });

    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("not ai");
    span.setAttribute("operation.name", "not ai stuff");
    span.end();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
  });

  it("should not export spans that do not pass the filter", () => {
    setupTraceProvider({
      Processor: OpenInferenceSimpleSpanProcessor,
      spanFilter: isOpenInferenceSpan,
    });
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("not ai");
    span.setAttribute("operation.name", "not ai stuff");
    span.end();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(0);
  });

  it("should not overwrite existing openinference.span.kind", () => {
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("ai.generateText.doGenerate");

    // Set the span kind manually first
    span.setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.CHAIN);
    span.setAttribute("operation.name", "ai.generateText.doGenerate");

    span.end();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);

    // The span kind should remain as CHAIN, not be overwritten to LLM
    expect(spans[0].attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.CHAIN,
    );
  });

  it("should rename root span to operation.name when present", () => {
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("original-name");
    span.setAttribute("operation.name", "ai.generateText my-function");
    span.setAttribute("ai.operationId", "ai.generateText");
    span.end();

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    expect(spans[0].name).toBe("ai.generateText my-function");
  });

  it("should remove ai.stream.* events from spans", () => {
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("ai.streamText.doStream");
    span.setAttribute("operation.name", "ai.streamText.doStream");
    span.addEvent("ai.stream.firstChunk", { "ai.stream.msToFirstChunk": 150 });
    span.addEvent("ai.stream.finish", {
      "ai.stream.msToFinish": 1200,
      "ai.usage.completionTokens": 42,
    });
    span.addEvent("other.event", { foo: "bar" });
    span.end();

    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
    // ai.stream.* events should be removed
    const eventNames = spans[0].events.map((e) => e.name);
    expect(eventNames).not.toContain("ai.stream.firstChunk");
    expect(eventNames).not.toContain("ai.stream.finish");
    // Non ai.stream.* events should be preserved
    expect(eventNames).toContain("other.event");
  });
});

describe("OpenInferenceBatchSpanProcessor", () => {
  beforeEach(() => {
    setupTraceProvider({ Processor: OpenInferenceBatchSpanProcessor });
  });

  test.each(generateVercelAttributeTestCases())(
    "should map %s",
    async (_name, { vercelFunctionName, vercelAttributes, expectedOpenInferenceAttributes }) => {
      const tracer = trace.getTracer("test-tracer");
      const span = tracer.startSpan(vercelFunctionName);
      const vercelAttributesWithOperationName = {
        ...vercelAttributes,
        "operation.name": vercelFunctionName,
      };

      span.setAttributes(vercelAttributesWithOperationName);
      span.end();
      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      // Check that expected attributes are present
      Object.entries(expectedOpenInferenceAttributes).forEach(([key, value]) => {
        expect(spans[0].attributes[key]).toEqual(value);
      });
    },
  );

  it("should not export non-AI spans", async () => {
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("ai.generateText");
    span.setAttribute("operation.name", "ai.generateText");
    span.end();
    const nonOpenInferenceSpan = tracer.startSpan("non-ai-span");
    nonOpenInferenceSpan.end();
    const spans = memoryExporter.getFinishedSpans();
    await processor.forceFlush();
    expect(spans.length).toBe(2);
  });

  it("should export all spans if there is no filter", async () => {
    setupTraceProvider({
      Processor: OpenInferenceBatchSpanProcessor,
    });

    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("not ai");
    span.setAttribute("operation.name", "not ai stuff");
    span.end();
    await processor.forceFlush();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);
  });

  it("should not export spans that do not pass the filter", async () => {
    setupTraceProvider({
      Processor: OpenInferenceBatchSpanProcessor,
      spanFilter: isOpenInferenceSpan,
    });
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("not ai");
    span.setAttribute("operation.name", "not ai stuff");
    span.end();
    await processor.forceFlush();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(0);
  });

  it("should not overwrite existing openinference.span.kind", async () => {
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("ai.generateText.doGenerate");

    // Set the span kind manually first
    span.setAttribute(SemanticConventions.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKind.CHAIN);
    span.setAttribute("operation.name", "ai.generateText.doGenerate");

    span.end();
    await processor.forceFlush();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);

    // The span kind should remain as CHAIN, not be overwritten to LLM
    expect(spans[0].attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
      OpenInferenceSpanKind.CHAIN,
    );
  });
});

describe("Trace aggregate behavior", () => {
  describe.each([
    ["OpenInferenceSimpleSpanProcessor", OpenInferenceSimpleSpanProcessor],
    ["OpenInferenceBatchSpanProcessor", OpenInferenceBatchSpanProcessor],
  ])("%s", (_name, Processor) => {
    beforeEach(() => {
      setupTraceProvider({ Processor });
    });
    afterEach(() => {
      trace.disable();
    });

    it("should propagate error status from child span to root span", async () => {
      const tracer = trace.getTracer("test-tracer");

      // Create a root AI SDK span
      const rootSpan = tracer.startSpan("ai.generateText");
      rootSpan.setAttribute("operation.name", "ai.generateText");

      // Create a child span within the root's context
      const rootContext = trace.setSpan(context.active(), rootSpan);
      const childSpan = tracer.startSpan("ai.generateText.doGenerate", undefined, rootContext);
      childSpan.setAttribute("operation.name", "ai.generateText.doGenerate");

      // Set error status on child span
      childSpan.setStatus({
        code: SpanStatusCode.ERROR,
        message: "Test error",
      });
      childSpan.end();

      // End root span (status should be UNSET initially, then set to ERROR)
      rootSpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();

      // Find the root span (no parent)
      const exportedRootSpan = spans.find((s) => s.parentSpanId == null);
      expect(exportedRootSpan).toBeDefined();
      expect(exportedRootSpan!.status.code).toBe(SpanStatusCode.ERROR);
      expect(exportedRootSpan!.status.message).toBe("Test error");
    });

    it("should set OK status on root span when no errors occur", async () => {
      const tracer = trace.getTracer("test-tracer");

      // Create a root AI SDK span
      const rootSpan = tracer.startSpan("ai.generateText");
      rootSpan.setAttribute("operation.name", "ai.generateText");

      // Create a child span within the root's context
      const rootContext = trace.setSpan(context.active(), rootSpan);
      const childSpan = tracer.startSpan("ai.generateText.doGenerate", undefined, rootContext);
      childSpan.setAttribute("operation.name", "ai.generateText.doGenerate");

      // End both spans without error
      childSpan.end();
      rootSpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();

      // Find the root span
      const exportedRootSpan = spans.find((s) => s.parentSpanId == null);
      expect(exportedRootSpan).toBeDefined();
      expect(exportedRootSpan!.status.code).toBe(SpanStatusCode.OK);

      // Find the child span - should also have OK status
      const exportedChildSpan = spans.find((s) => s.parentSpanId != null);
      expect(exportedChildSpan).toBeDefined();
      expect(exportedChildSpan!.status.code).toBe(SpanStatusCode.OK);
    });

    it("should detect error from ai.response.finishReason=error", async () => {
      const tracer = trace.getTracer("test-tracer");

      const rootSpan = tracer.startSpan("ai.generateText");
      rootSpan.setAttribute("operation.name", "ai.generateText");

      const rootContext = trace.setSpan(context.active(), rootSpan);
      const childSpan = tracer.startSpan("ai.generateText.doGenerate", undefined, rootContext);
      childSpan.setAttribute("operation.name", "ai.generateText.doGenerate");
      childSpan.setAttribute("ai.response.finishReason", "error");
      childSpan.end();

      rootSpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();

      const exportedRootSpan = spans.find((s) => s.parentSpanId == null);
      expect(exportedRootSpan).toBeDefined();
      expect(exportedRootSpan!.status.code).toBe(SpanStatusCode.ERROR);
    });

    it("should detect error from gen_ai.response.finish_reasons containing error", async () => {
      const tracer = trace.getTracer("test-tracer");

      const rootSpan = tracer.startSpan("ai.generateText");
      rootSpan.setAttribute("operation.name", "ai.generateText");

      const rootContext = trace.setSpan(context.active(), rootSpan);
      const childSpan = tracer.startSpan("ai.generateText.doGenerate", undefined, rootContext);
      childSpan.setAttribute("operation.name", "ai.generateText.doGenerate");
      childSpan.setAttribute("gen_ai.response.finish_reasons", ["stop", "error"]);
      childSpan.end();

      rootSpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();

      const exportedRootSpan = spans.find((s) => s.parentSpanId == null);
      expect(exportedRootSpan).toBeDefined();
      expect(exportedRootSpan!.status.code).toBe(SpanStatusCode.ERROR);
    });

    it("should not modify root span status if already set", async () => {
      const tracer = trace.getTracer("test-tracer");

      const rootSpan = tracer.startSpan("ai.generateText");
      rootSpan.setAttribute("operation.name", "ai.generateText");
      // Explicitly set OK status before any child errors
      rootSpan.setStatus({ code: SpanStatusCode.OK });

      const rootContext = trace.setSpan(context.active(), rootSpan);
      const childSpan = tracer.startSpan("ai.generateText.doGenerate", undefined, rootContext);
      childSpan.setAttribute("operation.name", "ai.generateText.doGenerate");
      childSpan.setStatus({
        code: SpanStatusCode.ERROR,
        message: "Test error",
      });
      childSpan.end();

      rootSpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();

      const exportedRootSpan = spans.find((s) => s.parentSpanId == null);
      expect(exportedRootSpan).toBeDefined();
      // Status should remain OK since it was explicitly set
      expect(exportedRootSpan!.status.code).toBe(SpanStatusCode.OK);
    });

    it("should only track AI SDK spans for aggregate state", async () => {
      const tracer = trace.getTracer("test-tracer");

      // Create a non-AI SDK span first
      const nonAISpan = tracer.startSpan("http-request");
      nonAISpan.setAttribute("http.method", "GET");

      // Create an AI SDK root span
      const rootSpan = tracer.startSpan("ai.generateText");
      rootSpan.setAttribute("operation.name", "ai.generateText");

      const rootContext = trace.setSpan(context.active(), rootSpan);
      const childSpan = tracer.startSpan("ai.generateText.doGenerate", undefined, rootContext);
      childSpan.setAttribute("operation.name", "ai.generateText.doGenerate");
      childSpan.setStatus({
        code: SpanStatusCode.ERROR,
        message: "Test error",
      });
      childSpan.end();

      rootSpan.end();

      // End the non-AI span - this should not affect aggregate tracking
      nonAISpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();

      // Find the AI root span
      const exportedRootSpan = spans.find(
        (s) => s.parentSpanId == null && s.attributes["operation.name"] === "ai.generateText",
      );
      expect(exportedRootSpan).toBeDefined();
      expect(exportedRootSpan!.status.code).toBe(SpanStatusCode.ERROR);
    });

    it("should not set status on non-AI SDK root spans", async () => {
      const tracer = trace.getTracer("test-tracer");

      // Create a non-AI SDK root span
      const rootSpan = tracer.startSpan("http-request");
      rootSpan.setAttribute("http.method", "GET");

      const rootContext = trace.setSpan(context.active(), rootSpan);
      const childSpan = tracer.startSpan("db-query", undefined, rootContext);
      childSpan.setStatus({ code: SpanStatusCode.ERROR, message: "DB error" });
      childSpan.end();

      rootSpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();

      const exportedRootSpan = spans.find((s) => s.parentSpanId == null);
      expect(exportedRootSpan).toBeDefined();
      // Status should remain UNSET for non-AI SDK spans
      expect(exportedRootSpan!.status.code).toBe(SpanStatusCode.UNSET);
    });

    it("should handle spans detected via gen_ai.* attributes", async () => {
      const tracer = trace.getTracer("test-tracer");

      // Create a root span with gen_ai.* attributes (AI SDK v6 style)
      const rootSpan = tracer.startSpan("chat");
      rootSpan.setAttribute("gen_ai.system", "openai");

      const rootContext = trace.setSpan(context.active(), rootSpan);
      const childSpan = tracer.startSpan("llm-call", undefined, rootContext);
      childSpan.setAttribute("gen_ai.request.model", "gpt-4");
      childSpan.setStatus({ code: SpanStatusCode.ERROR, message: "API error" });
      childSpan.end();

      rootSpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();

      const exportedRootSpan = spans.find((s) => s.parentSpanId == null);
      expect(exportedRootSpan).toBeDefined();
      expect(exportedRootSpan!.status.code).toBe(SpanStatusCode.ERROR);
    });

    it("should clear aggregate state on shutdown", async () => {
      const tracer = trace.getTracer("test-tracer");

      // Start a span but don't end it (simulating abandoned span)
      const span = tracer.startSpan("ai.generateText");
      span.setAttribute("operation.name", "ai.generateText");

      // Shutdown should clear internal state without error
      await processor.shutdown();

      // The processor should not throw or leak memory
      // We can't easily verify internal state, but at least verify no errors occur
      expect(true).toBe(true);
    });

    it("should clear aggregate state on forceFlush", async () => {
      const tracer = trace.getTracer("test-tracer");

      // Start a span but don't end it (simulating abandoned span)
      const span = tracer.startSpan("ai.generateText");
      span.setAttribute("operation.name", "ai.generateText");

      // forceFlush should clear internal state without error
      await processor.forceFlush();

      // Create and complete a new span to verify processor still works
      const newSpan = tracer.startSpan("ai.generateText");
      newSpan.setAttribute("operation.name", "ai.generateText");
      newSpan.end();

      await processor.forceFlush();
      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBeGreaterThan(0);
    });
  });
});

describe.each([
  ["OpenInferenceSimpleSpanProcessor", OpenInferenceSimpleSpanProcessor],
  ["OpenInferenceBatchSpanProcessor", OpenInferenceBatchSpanProcessor],
] as [string, typeof OpenInferenceSimpleSpanProcessor | typeof OpenInferenceBatchSpanProcessor][])(
  "%s — reparentOrphanedSpans",
  (_name, Processor) => {
    const build = (reparentOrphanedSpans?: boolean) => {
      const exporter = new InMemorySpanExporter();
      const proc = new Processor({
        exporter,
        spanFilter: isOpenInferenceSpan,
        reparentOrphanedSpans,
      });
      const provider = new BasicTracerProvider({ spanProcessors: [proc] });
      return { exporter, provider, tracer: provider.getTracer("test") };
    };

    // Emits: GET /chat (non-AI root) -> ai.generateText -> ai.generateText.doGenerate
    const emitHttpWrappedAI = (tracer: ReturnType<typeof build>["tracer"]) => {
      const http = tracer.startSpan("GET /chat", {
        attributes: { "http.request.method": "GET", "http.route": "/chat" },
      });
      const top = tracer.startSpan(
        "ai.generateText",
        { attributes: { "operation.name": "ai.generateText" } },
        trace.setSpan(context.active(), http),
      );
      const llm = tracer.startSpan(
        "ai.generateText.doGenerate",
        { attributes: { "operation.name": "ai.generateText.doGenerate" } },
        trace.setSpan(context.active(), top),
      );
      llm.end();
      top.end();
      http.end();
      return {
        httpId: http.spanContext().spanId,
        topId: top.spanContext().spanId,
        llmId: llm.spanContext().spanId,
      };
    };

    it("re-roots the top AI span when its non-AI parent is filtered out", async () => {
      const { exporter, provider, tracer } = build(true);
      const { topId } = emitHttpWrappedAI(tracer);

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      // Non-AI HTTP span filtered out; AI spans exported.
      expect(spans.find((s) => s.name === "GET /chat")).toBeUndefined();
      const top = spans.find((s) => s.name === "ai.generateText");
      expect(top?.parentSpanId).toBeUndefined(); // re-rooted
      // Subtree intact: the LLM child stays parented to the (now-root) top AI span.
      const llm = spans.find((s) => s.name === "ai.generateText.doGenerate");
      expect(llm?.parentSpanId).toBe(topId);
    });

    it("does not mutate the caller's live span object (re-roots only the exported span)", async () => {
      const { exporter, provider, tracer } = build(true);

      const http = tracer.startSpan("GET /chat", {
        attributes: { "http.request.method": "GET", "http.route": "/chat" },
      });
      const httpId = http.spanContext().spanId;
      const top = tracer.startSpan(
        "ai.generateText",
        { attributes: { "operation.name": "ai.generateText" } },
        trace.setSpan(context.active(), http),
      );

      // The live Span object the host runtime owns must keep its real parent
      // linkage: re-rooting for export must NOT mutate the caller's span in
      // place. Host runtimes that manage span lifecycle around that parent
      // (e.g. Vercel eve's durable workflow) otherwise get driven into
      // post-end span operations — a flood of "Operation attempted on ended
      // Span" warnings. (The parent is exposed as `parentSpanId` on 1.x spans
      // and `parentSpanContext` on 2.x; the re-rooting clears both on export.)
      const liveTop = top as unknown as {
        parentSpanId?: string;
        parentSpanContext?: { spanId: string };
      };
      const liveParentId = liveTop.parentSpanId ?? liveTop.parentSpanContext?.spanId;
      expect(liveParentId).toBe(httpId);

      top.end();
      http.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      // ...but the EXPORTED span is still re-rooted to be a clean trace root.
      const exportedTop = spans.find((s) => s.name === "ai.generateText");
      expect(exportedTop?.parentSpanId).toBeUndefined();
    });

    it("leaves the top AI span orphaned when reparentOrphanedSpans is off (default)", async () => {
      const { exporter, provider, tracer } = build(); // default: off
      const { httpId } = emitHttpWrappedAI(tracer);

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      expect(spans.find((s) => s.name === "GET /chat")).toBeUndefined();
      const top = spans.find((s) => s.name === "ai.generateText");
      // Still points at the filtered-out HTTP parent → orphaned.
      expect(top?.parentSpanId).toBe(httpId);
    });

    it("re-roots multiple sibling AI spans under one non-AI parent", async () => {
      const { exporter, provider, tracer } = build(true);

      const http = tracer.startSpan("GET /batch", {
        attributes: { "http.request.method": "POST", "http.route": "/batch" },
      });
      const httpCtx = trace.setSpan(context.active(), http);
      const a = tracer.startSpan(
        "ai.generateText",
        { attributes: { "operation.name": "ai.generateText turn-a" } },
        httpCtx,
      );
      const b = tracer.startSpan(
        "ai.generateText",
        { attributes: { "operation.name": "ai.generateText turn-b" } },
        httpCtx,
      );
      a.end();
      b.end();
      http.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      // Re-rooted root spans are renamed to their operation.name (with functionId suffix).
      const aiSpans = spans.filter((s) => s.name.startsWith("ai.generateText"));
      expect(aiSpans).toHaveLength(2);
      // Both sibling AI spans are re-rooted (neither is orphaned).
      expect(aiSpans.every((s) => s.parentSpanId == null)).toBe(true);
    });

    it("does not re-root an AI span nested under an AI parent", async () => {
      const { exporter, provider, tracer } = build(true);

      const top = tracer.startSpan("ai.streamText", {
        attributes: { "operation.name": "ai.streamText" },
      });
      const llm = tracer.startSpan(
        "ai.streamText.doStream",
        { attributes: { "operation.name": "ai.streamText.doStream" } },
        trace.setSpan(context.active(), top),
      );
      llm.end();
      top.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      const child = spans.find((s) => s.name === "ai.streamText.doStream");
      // Parent is an AI span → the child keeps its parent (subtree preserved).
      expect(child?.parentSpanId).toBe(top.spanContext().spanId);
    });

    it("preserves a kind-less AI wrapper root (e.g. ai.eve.turn) as a promoted AGENT root", async () => {
      const { exporter, provider, tracer } = build(true);

      // Non-AI workflow/HTTP span that the filter drops.
      const workflow = tracer.startSpan("vercel.workflow");
      // A framework wrapper with an ai.* operation name the kind map does NOT recognize,
      // so conversion leaves it kind-less. (Matched by shape, not by name.)
      const turn = tracer.startSpan(
        "ai.eve.turn",
        { attributes: { "operation.name": "ai.eve.turn" } },
        trace.setSpan(context.active(), workflow),
      );
      // A recognized LLM child.
      const llm = tracer.startSpan(
        "ai.streamText.doStream",
        { attributes: { "operation.name": "ai.streamText.doStream" } },
        trace.setSpan(context.active(), turn),
      );
      llm.end();
      turn.end();
      workflow.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      // Non-AI workflow span is filtered out.
      expect(spans.find((s) => s.name === "vercel.workflow")).toBeUndefined();

      // The kind-less wrapper is re-rooted AND promoted to an AGENT so it survives the filter.
      const promoted = spans.find((s) => s.name === "ai.eve.turn");
      expect(promoted).toBeDefined();
      expect(promoted?.parentSpanId).toBeUndefined();
      expect(promoted?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.AGENT,
      );

      // The LLM child stays attached to the promoted root.
      const child = spans.find((s) => s.name === "ai.streamText.doStream");
      expect(child?.parentSpanId).toBe(promoted?.spanContext().spanId);
    });

    it("promotes a kind-less AI wrapper that is a natural root (no parent) to an AGENT root", async () => {
      const { exporter, provider, tracer } = build(true);

      // A kind-less AI-like wrapper started as a top-level root (no parent at all) — e.g. a
      // framework "turn" span that isn't nested under an HTTP/workflow span. It is not
      // re-rooted (already a root), but must still be promoted to AGENT so it survives the
      // filter instead of being dropped as kind-less.
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "ai.eve.turn" },
      });
      const llm = tracer.startSpan(
        "ai.streamText.doStream",
        { attributes: { "operation.name": "ai.streamText.doStream" } },
        trace.setSpan(context.active(), turn),
      );
      llm.end();
      turn.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      const promoted = spans.find((s) => s.name === "ai.eve.turn");
      expect(promoted).toBeDefined();
      expect(promoted?.parentSpanId).toBeUndefined();
      expect(promoted?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.AGENT,
      );
      // The LLM child stays attached to the promoted root.
      const child = spans.find((s) => s.name === "ai.streamText.doStream");
      expect(child?.parentSpanId).toBe(promoted?.spanContext().spanId);
    });

    // Real-world eve shape: the per-turn wrapper's `operation.name` is "eve" (NOT ai.*),
    // it has no gen_ai.* attributes, and is marked only by the AI SDK telemetry attribute
    // `ai.telemetry.functionId`. It must still be recognized as AI so it is kept as the single
    // root with the agent subtree nested under it — otherwise its AI children are each
    // re-rooted into separate roots (the orphan-spans bug seen in eve traces).
    it("recognizes a wrapper marked only by ai.telemetry.* attrs (real ai.eve.turn shape)", async () => {
      const { exporter, provider, tracer } = build(true);

      // Non-AI workflow span the filter drops.
      const workflow = tracer.startSpan("step.execute", {
        attributes: { "operation.name": "workflow" },
      });
      // The eve turn wrapper as actually emitted: operation.name "eve", no gen_ai.*, only
      // an ai.telemetry.* marker. The kind map does not recognize it (kind-less).
      const turn = tracer.startSpan(
        "ai.eve.turn",
        {
          attributes: {
            "operation.name": "eve",
            "ai.telemetry.functionId": "weatherbot",
          },
        },
        trace.setSpan(context.active(), workflow),
      );
      const turnId = turn.spanContext().spanId;
      // A real AI SDK child (gen_ai invoke_agent) under the turn.
      const agent = tracer.startSpan(
        "invoke_agent",
        { attributes: { "gen_ai.operation.name": "invoke_agent" } },
        trace.setSpan(context.active(), turn),
      );
      const agentId = agent.spanContext().spanId;
      agent.end();
      turn.end();
      workflow.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      // The turn wrapper survives as the single promoted AGENT root. (Re-rooted AI roots are
      // renamed to their operation.name, so match by spanId rather than name.)
      const promoted = spans.find((s) => s.spanContext().spanId === turnId);
      expect(promoted).toBeDefined();
      expect(promoted?.parentSpanId).toBeUndefined();
      expect(promoted?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.AGENT,
      );

      // The agent child stays nested under the turn — NOT re-rooted into its own root.
      const child = spans.find((s) => s.spanContext().spanId === agentId);
      expect(child?.parentSpanId).toBe(turnId);
    });

    // The root-span rename surfaces the AI SDK operation.name (e.g. "ai.generateText <fnId>").
    // A framework wrapper like ai.eve.turn has operation.name "eve" — renaming would clobber
    // the meaningful span name with "eve". Only rename when operation.name is itself ai.*.
    it("does not rename a re-rooted wrapper whose operation.name is not ai.* (keeps ai.eve.turn)", async () => {
      const { exporter, provider, tracer } = build(true);

      const workflow = tracer.startSpan("step.execute", {
        attributes: { "operation.name": "workflow" },
      });
      const turn = tracer.startSpan(
        "ai.eve.turn",
        {
          attributes: {
            "operation.name": "eve",
            "ai.telemetry.functionId": "weatherbot",
          },
        },
        trace.setSpan(context.active(), workflow),
      );
      const turnId = turn.spanContext().spanId;
      turn.end();
      workflow.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      const promoted = spans.find((s) => s.spanContext().spanId === turnId);
      // Name is preserved — NOT clobbered to the "eve" operation.name.
      expect(promoted?.name).toBe("ai.eve.turn");
    });

    // Real eve topology: ONE ai.eve.turn wrapper under a non-AI workflow span, with MULTIPLE
    // invoke_agent children (one per agent step). The wrapper must become the single root with
    // ALL children nested under it — not split into one root per child (the orphan-spans bug).
    it("keeps multiple AI children under one re-rooted wrapper (single root, no split)", async () => {
      const { exporter, provider, tracer } = build(true);

      const workflow = tracer.startSpan("step.execute", {
        attributes: { "operation.name": "workflow" },
      });
      const turn = tracer.startSpan(
        "ai.eve.turn",
        { attributes: { "operation.name": "eve", "ai.telemetry.functionId": "weatherbot" } },
        trace.setSpan(context.active(), workflow),
      );
      const turnId = turn.spanContext().spanId;
      const turnCtx = trace.setSpan(context.active(), turn);
      const agentA = tracer.startSpan(
        "invoke_agent",
        { attributes: { "gen_ai.operation.name": "invoke_agent" } },
        turnCtx,
      );
      const agentB = tracer.startSpan(
        "invoke_agent",
        { attributes: { "gen_ai.operation.name": "invoke_agent" } },
        turnCtx,
      );
      const aId = agentA.spanContext().spanId;
      const bId = agentB.spanContext().spanId;
      agentA.end();
      agentB.end();
      turn.end();
      workflow.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      // Exactly one root: the turn wrapper.
      const roots = spans.filter((s) => s.parentSpanId == null);
      expect(roots).toHaveLength(1);
      expect(roots[0]?.spanContext().spanId).toBe(turnId);
      // Both invoke_agent children stay nested under it — neither is re-rooted.
      expect(spans.find((s) => s.spanContext().spanId === aId)?.parentSpanId).toBe(turnId);
      expect(spans.find((s) => s.spanContext().spanId === bId)?.parentSpanId).toBe(turnId);
    });

    // Across an async/durable boundary (e.g. eve workflow steps) a later AI span starts with
    // its parent represented as a bare, non-recording SpanContext — the parent's attributes are
    // not inspectable, even though its spanId is correct and the parent is itself exported.
    // The re-root check must NOT treat "can't inspect parent" as "parent is non-AI", or it
    // orphans the child off an exported AI parent.
    it("does not re-root an AI span whose parent is a non-recording (propagated) span", async () => {
      const { exporter, provider, tracer } = build(true);

      // The eve turn wrapper — a real recording AI span that is exported as the root.
      const turn = tracer.startSpan("ai.eve.turn", {
        attributes: { "operation.name": "eve", "ai.telemetry.functionId": "weatherbot" },
      });
      const turnId = turn.spanContext().spanId;

      // Simulate the boundary: the child starts under a context that carries only the turn's
      // SpanContext (a non-recording span — no attributes), as happens after a workflow hop.
      const propagated = trace.setSpanContext(context.active(), turn.spanContext());
      const agent = tracer.startSpan(
        "invoke_agent",
        { attributes: { "gen_ai.operation.name": "invoke_agent" } },
        propagated,
      );
      const agentId = agent.spanContext().spanId;
      agent.end();
      turn.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      // The child stays attached to the exported turn — NOT re-rooted into a second root.
      const child = spans.find((s) => s.spanContext().spanId === agentId);
      expect(child?.parentSpanId).toBe(turnId);
    });

    // Proves the AGENT promotion is keyed off the `ai.` shape, NOT the `ai.eve.turn` name:
    // arbitrary unrecognized ai.* wrappers are promoted too, and a non-AI-prefixed wrapper
    // is not.
    it.each([
      ["ai.eve.turn", true],
      ["ai.workflow.run", true],
      ["ai.customAgent.session", true],
      ["acme.workflow.run", false], // not ai.*-prefixed → not AI-like → not promoted
    ] as [string, boolean][])(
      "promotes kind-less wrapper %s only when it is AI-like (name-agnostic)",
      async (wrapperName, expectPromoted) => {
        const { exporter, provider, tracer } = build(true);

        const outer = tracer.startSpan("vercel.workflow");
        const wrapper = tracer.startSpan(
          wrapperName,
          { attributes: { "operation.name": wrapperName } },
          trace.setSpan(context.active(), outer),
        );
        tracer
          .startSpan(
            "ai.streamText.doStream",
            { attributes: { "operation.name": "ai.streamText.doStream" } },
            trace.setSpan(context.active(), wrapper),
          )
          .end();
        wrapper.end();
        outer.end();

        await provider.forceFlush();
        const spans = exporter.getFinishedSpans();
        await provider.shutdown();

        const promoted = spans.find((s) => s.name === wrapperName);
        if (expectPromoted) {
          expect(promoted?.parentSpanId).toBeUndefined();
          expect(promoted?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
            OpenInferenceSpanKind.AGENT,
          );
        } else {
          // Not AI-like → never re-rooted or promoted → dropped by the filter.
          expect(promoted).toBeUndefined();
        }
      },
    );

    it("promotes multiple sibling kind-less AI wrappers, each to its own AGENT root", async () => {
      const { exporter, provider, tracer } = build(true);

      const http = tracer.startSpan("GET /chat", { attributes: { "http.route": "/chat" } });
      const httpCtx = trace.setSpan(context.active(), http);
      // Two independent unrecognized ai.* turn wrappers in one trace (non-eve names).
      for (const id of ["a", "b"]) {
        const turn = tracer.startSpan(
          "ai.agent.turn",
          { attributes: { "operation.name": `ai.agent.turn ${id}` } },
          httpCtx,
        );
        tracer
          .startSpan(
            "ai.streamText.doStream",
            { attributes: { "operation.name": `ai.streamText.doStream ${id}` } },
            trace.setSpan(context.active(), turn),
          )
          .end();
        turn.end();
      }
      http.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      // Both wrappers are promoted to parentless AGENT roots (renamed to operation.name).
      const wrappers = spans.filter((s) => s.name.startsWith("ai.agent.turn"));
      expect(wrappers).toHaveLength(2);
      expect(
        wrappers.every(
          (s) =>
            s.parentSpanId == null &&
            s.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] ===
              OpenInferenceSpanKind.AGENT,
        ),
      ).toBe(true);
      // Each LLM child is attached to one of the promoted roots (none orphaned).
      const wrapperIds = new Set(wrappers.map((s) => s.spanContext().spanId));
      const llms = spans.filter((s) => s.name.startsWith("ai.streamText.doStream"));
      expect(llms).toHaveLength(2);
      expect(llms.every((s) => s.parentSpanId != null && wrapperIds.has(s.parentSpanId))).toBe(
        true,
      );
    });

    it("re-roots a recognized AI span (ai.embed) while keeping its mapped kind (CHAIN, not AGENT)", async () => {
      const { exporter, provider, tracer } = build(true);

      // Non-AI parent the filter drops.
      const http = tracer.startSpan("GET /embed", { attributes: { "http.route": "/embed" } });
      // `ai.embed` IS recognized by the kind map → CHAIN. The AGENT promotion only ever
      // fires on kind-less wrappers, so re-rooting must leave this span's mapped kind intact.
      const embed = tracer.startSpan(
        "ai.embed",
        { attributes: { "operation.name": "ai.embed" } },
        trace.setSpan(context.active(), http),
      );
      embed.end();
      http.end();

      await provider.forceFlush();
      const spans = exporter.getFinishedSpans();
      await provider.shutdown();

      expect(spans.find((s) => s.name === "GET /embed")).toBeUndefined();
      const reRooted = spans.find((s) => s.name.startsWith("ai.embed"));
      expect(reRooted?.parentSpanId).toBeUndefined(); // re-rooted
      // Kept CHAIN from the kind map — NOT overridden to AGENT by promotion.
      expect(reRooted?.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND]).toBe(
        OpenInferenceSpanKind.CHAIN,
      );
    });
  },
);
