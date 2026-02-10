import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { Attributes, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";

import {
  isOpenInferenceSpan,
  OpenInferenceBatchSpanProcessor,
  OpenInferenceSimpleSpanProcessor,
  SpanFilter,
} from "../src";
import { VercelSDKFunctionNameToSpanKindMap } from "../src/constants";
import { VercelAISemanticConventions } from "../src/VercelAISemanticConventions";

import embedDoEmbedFixture from "./__fixtures__/v6-spans/ai-embed-doEmbed.json";
import generateObjectDoGenerateFixture from "./__fixtures__/v6-spans/ai-generateObject-doGenerate.json";
// Import real AI SDK v6 fixtures
import generateTextDoGenerateFixture from "./__fixtures__/v6-spans/ai-generateText-doGenerate.json";
import streamTextDoStreamFixture from "./__fixtures__/v6-spans/ai-streamText-doStream.json";

import { afterEach, beforeEach, describe, expect, it, test } from "vitest";

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
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
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
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
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
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.EMBEDDING,
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
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.EMBEDDING,
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
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
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
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.CHAIN,
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
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
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
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.LLM,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.LLM,
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
          { toolName: "test-tool-1", args: { test1: "test-1" } },
          { toolName: "test-tool-2", input: { test2: "test-2" } },
        ]),
      },
      expectedOpenInferenceAttributes: {
        [`${firstOutputMessageToolPrefix}.0.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
          "test-tool-1",
        [`${firstOutputMessageToolPrefix}.0.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
          JSON.stringify({ test1: "test-1" }),
        [`${firstOutputMessageToolPrefix}.1.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
          "test-tool-2",
        [`${firstOutputMessageToolPrefix}.1.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
          JSON.stringify({ test2: "test-2" }),
        [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
          "assistant",
        [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
          OpenInferenceSpanKind.TOOL,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.EMBEDDING,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.EMBEDDING,
        },
      },
    ],
    [
      `${VercelAISemanticConventions.EMBEDDING_VECTOR} to ${SemanticConventions.EMBEDDING_VECTOR}`,
      {
        vercelFunctionName: "ai.embedMany.doEmbed",
        vercelAttributes: {
          [VercelAISemanticConventions.EMBEDDING_VECTOR]: JSON.stringify([
            1, 2,
          ]),
        },
        expectedOpenInferenceAttributes: {
          [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
            [1, 2],
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.EMBEDDING,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.EMBEDDING,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.TOOL,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.TOOL,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.TOOL,
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
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
            OpenInferenceSpanKind.TOOL,
        },
      },
    ],
  );

  return testCases;
};

let traceProvider = new BasicTracerProvider();
let memoryExporter = new InMemorySpanExporter();
let processor:
  | OpenInferenceSimpleSpanProcessor
  | OpenInferenceBatchSpanProcessor;

function setupTraceProvider({
  Processor,
  spanFilter,
}: {
  Processor:
    | typeof OpenInferenceBatchSpanProcessor
    | typeof OpenInferenceSimpleSpanProcessor;
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
      expect(
        spans[0].attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND],
      ).toBe(spanKind);
      memoryExporter.reset();
    });
  });

  test.each(generateVercelAttributeTestCases())(
    "should map %s",
    (
      _name,
      { vercelFunctionName, vercelAttributes, expectedOpenInferenceAttributes },
    ) => {
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
      Object.entries(expectedOpenInferenceAttributes).forEach(
        ([key, value]) => {
          expect(spans[0].attributes[key]).toEqual(value);
        },
      );
    },
  );

  test.each(generateV6FixtureTestCases())(
    "should correctly process %s",
    (
      _name,
      { vercelFunctionName, vercelAttributes, expectedOpenInferenceAttributes },
    ) => {
      const tracer = trace.getTracer("test-tracer");
      const span = tracer.startSpan(vercelFunctionName);
      span.setAttributes(vercelAttributes);
      span.end();
      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      // Check that expected attributes are present
      Object.entries(expectedOpenInferenceAttributes).forEach(
        ([key, value]) => {
          expect(spans[0].attributes[key]).toEqual(value);
        },
      );
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
    span.setAttribute(
      SemanticConventions.OPENINFERENCE_SPAN_KIND,
      OpenInferenceSpanKind.CHAIN,
    );
    span.setAttribute("operation.name", "ai.generateText.doGenerate");

    span.end();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);

    // The span kind should remain as CHAIN, not be overwritten to LLM
    expect(
      spans[0].attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND],
    ).toBe(OpenInferenceSpanKind.CHAIN);
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
});

describe("OpenInferenceBatchSpanProcessor", () => {
  beforeEach(() => {
    setupTraceProvider({ Processor: OpenInferenceBatchSpanProcessor });
  });

  test.each(generateVercelAttributeTestCases())(
    "should map %s",
    async (
      _name,
      { vercelFunctionName, vercelAttributes, expectedOpenInferenceAttributes },
    ) => {
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
      Object.entries(expectedOpenInferenceAttributes).forEach(
        ([key, value]) => {
          expect(spans[0].attributes[key]).toEqual(value);
        },
      );
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
    span.setAttribute(
      SemanticConventions.OPENINFERENCE_SPAN_KIND,
      OpenInferenceSpanKind.CHAIN,
    );
    span.setAttribute("operation.name", "ai.generateText.doGenerate");

    span.end();
    await processor.forceFlush();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(1);

    // The span kind should remain as CHAIN, not be overwritten to LLM
    expect(
      spans[0].attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND],
    ).toBe(OpenInferenceSpanKind.CHAIN);
  });
});
