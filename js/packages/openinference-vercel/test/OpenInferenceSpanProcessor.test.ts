import { Attributes, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
} from "@opentelemetry/sdk-trace-base";
import { VercelSDKFunctionNameToSpanKindMap } from "../src/constants";
import {
  isOpenInferenceSpan,
  OpenInferenceBatchSpanProcessor,
  OpenInferenceSimpleSpanProcessor,
  SpanFilter,
} from "../src";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  AISemanticConventions,
  AISemanticConventionsList,
} from "../src/AISemanticConventions";
import { assertUnreachable } from "../src/typeUtils";

type SpanProcessorTestCase = [
  string,
  {
    vercelFunctionName: string;
    vercelAttributes: Attributes;
    addedOpenInferenceAttributes: Attributes;
  },
];

const generateVercelAttributeTestCases = (): SpanProcessorTestCase[] => {
  const testCases: SpanProcessorTestCase[] = [];
  AISemanticConventionsList.map((vercelSemanticConvention) => {
    switch (vercelSemanticConvention) {
      case AISemanticConventions.MODEL_ID:
        testCases.push(
          [
            `${vercelSemanticConvention} (LLMs) to ${SemanticConventions.LLM_MODEL_NAME} for LLM`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: { [vercelSemanticConvention]: "test-llm" },
              addedOpenInferenceAttributes: {
                [SemanticConventions.LLM_MODEL_NAME]: "test-llm",
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
              },
            },
          ],
          [
            `${vercelSemanticConvention} (embeddings) to ${SemanticConventions.EMBEDDING_MODEL_NAME} for embeddings`,
            {
              vercelFunctionName: "ai.embed.doEmbed",
              vercelAttributes: {
                [vercelSemanticConvention]: "test-embedding-model",
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.EMBEDDING_MODEL_NAME]:
                  "test-embedding-model",
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.EMBEDDING,
              },
            },
          ],
        );
        break;
      case AISemanticConventions.METADATA:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.METADATA}`,
          {
            vercelFunctionName: "ai.generateText.doGenerate",
            vercelAttributes: {
              [`${vercelSemanticConvention}.key1`]: "value1",
              [`${vercelSemanticConvention}.key2`]: "value2",
            },
            addedOpenInferenceAttributes: {
              [`${SemanticConventions.METADATA}.key1`]: "value1",
              [`${SemanticConventions.METADATA}.key2`]: "value2",
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.LLM,
            },
          },
        ]);
        break;
      case AISemanticConventions.SETTINGS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.LLM_INVOCATION_PARAMETERS}, accumulating and stringifying all settings`,
          {
            vercelFunctionName: "ai.generateText.doGenerate",
            vercelAttributes: {
              [`${vercelSemanticConvention}.key1`]: "value1",
              [`${vercelSemanticConvention}.key2`]: "value2",
            },
            addedOpenInferenceAttributes: {
              [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify({
                key1: "value1",
                key2: "value2",
              }),
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.LLM,
            },
          },
        ]);
        break;
      case AISemanticConventions.TOKEN_COUNT_COMPLETION:
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.LLM_TOKEN_COUNT_COMPLETION}`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: 10,
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 10,
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
              },
            },
          ],
          [
            `${vercelSemanticConvention} to nothing if a chain span`,
            {
              vercelFunctionName: "ai.generateText",
              vercelAttributes: {
                [vercelSemanticConvention]: 10,
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.CHAIN,
              },
            },
          ],
        );
        break;
      case AISemanticConventions.TOKEN_COUNT_PROMPT:
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: 10,
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 10,
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
              },
            },
          ],
          [
            `${vercelSemanticConvention} to nothing if a chain span`,
            {
              vercelFunctionName: "ai.generateText",
              vercelAttributes: {
                [vercelSemanticConvention]: 10,
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.CHAIN,
              },
            },
          ],
        );
        break;
      case AISemanticConventions.RESULT_TEXT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.OUTPUT_VALUE} with MIME type ${MimeType.TEXT}`,
          {
            vercelFunctionName: "ai.generateText.doGenerate",
            vercelAttributes: {
              [vercelSemanticConvention]: "hello",
            },
            addedOpenInferenceAttributes: {
              [SemanticConventions.OUTPUT_VALUE]: "hello",
              [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.LLM,
            },
          },
        ]);
        break;
      case AISemanticConventions.RESULT_OBJECT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.OUTPUT_VALUE} with MIME type ${MimeType.JSON}`,
          {
            vercelFunctionName: "ai.generateObject.doGenerate",
            vercelAttributes: {
              [vercelSemanticConvention]: JSON.stringify({ key: "value" }),
            },
            addedOpenInferenceAttributes: {
              [SemanticConventions.OUTPUT_VALUE]: JSON.stringify({
                key: "value",
              }),
              [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.LLM,
            },
          },
        ]);
        break;
      case AISemanticConventions.RESULT_TOOL_CALLS: {
        const firstOutputMessageToolPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}`;
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.MESSAGE_TOOL_CALLS} on ${SemanticConventions.LLM_OUTPUT_MESSAGES}`,
          {
            vercelFunctionName: "ai.toolCall",
            vercelAttributes: {
              [vercelSemanticConvention]: JSON.stringify([
                { toolName: "test-tool-1", args: { test1: "test-1" } },
                { toolName: "test-tool-2", args: { test2: "test-2" } },
              ]),
            },
            addedOpenInferenceAttributes: {
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
        break;
      }
      case AISemanticConventions.PROMPT:
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.INPUT_VALUE} with MIME type ${MimeType.TEXT} for normal strings`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: "hello",
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.INPUT_VALUE]: "hello",
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
              },
            },
          ],
          [
            `${vercelSemanticConvention} to ${SemanticConventions.INPUT_VALUE} with MIME type ${MimeType.JSON} for JSON object strings`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: JSON.stringify({}),
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.INPUT_VALUE]: "{}",
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
              },
            },
          ],
        );
        break;
      case AISemanticConventions.PROMPT_MESSAGES: {
        const firstInputMessageContentsPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}`;
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.LLM_INPUT_MESSAGES} with role and content for string content messages`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: JSON.stringify([
                  { role: "assistant", content: "hello" },
                  { role: "user", content: "world" },
                ]),
              },
              addedOpenInferenceAttributes: {
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
            `${vercelSemanticConvention} to ${SemanticConventions.LLM_INPUT_MESSAGES} with role and content for object content messages`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: JSON.stringify([
                  {
                    role: "assistant",
                    content: [
                      { type: "text", text: "hello" },
                      { type: "image", image: "image.com" },
                    ],
                  },
                ]),
              },
              addedOpenInferenceAttributes: {
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
                  "assistant",
                [`${firstInputMessageContentsPrefix}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
                  "text",
                [`${firstInputMessageContentsPrefix}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`]:
                  "hello",
                [`${firstInputMessageContentsPrefix}.0.${SemanticConventions.MESSAGE_CONTENT_IMAGE}`]:
                  undefined,
                [`${firstInputMessageContentsPrefix}.1.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
                  "image",
                [`${firstInputMessageContentsPrefix}.1.${SemanticConventions.MESSAGE_CONTENT_IMAGE}`]:
                  "image.com",
                [`${firstInputMessageContentsPrefix}.1.${SemanticConventions.MESSAGE_CONTENT_TEXT}`]:
                  undefined,
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
              },
            },
          ],
        );
        break;
      }
      case AISemanticConventions.EMBEDDING_TEXT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_TEXT}`,
          {
            vercelFunctionName: "ai.embed.doEmbed",
            vercelAttributes: {
              [vercelSemanticConvention]: "hello",
            },
            addedOpenInferenceAttributes: {
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
                "hello",
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.EMBEDDING,
            },
          },
        ]);
        break;
      case AISemanticConventions.EMBEDDING_TEXTS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_TEXT} for multiple texts`,
          {
            vercelFunctionName: "ai.embedMany.doEmbed",
            vercelAttributes: {
              [vercelSemanticConvention]: ["hello", "world"],
            },
            addedOpenInferenceAttributes: {
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
                "hello",
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_TEXT}`]:
                "world",
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.EMBEDDING,
            },
          },
        ]);
        break;
      case AISemanticConventions.EMBEDDING_VECTOR:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_VECTOR}`,
          {
            vercelFunctionName: "ai.embedMany.doEmbed",
            vercelAttributes: {
              [vercelSemanticConvention]: JSON.stringify([1, 2]),
            },
            addedOpenInferenceAttributes: {
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
                [1, 2],
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.EMBEDDING,
            },
          },
        ]);
        break;
      case AISemanticConventions.EMBEDDING_VECTORS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_VECTOR} for multiple vectors`,
          {
            vercelFunctionName: "ai.embedMany.doEmbed",
            vercelAttributes: {
              [vercelSemanticConvention]: ["[1, 2]", "[3, 4]"],
            },
            addedOpenInferenceAttributes: {
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
                [1, 2],
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_VECTOR}`]:
                [3, 4],
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.EMBEDDING,
            },
          },
        ]);
        break;
      case AISemanticConventions.TOOL_CALL_NAME:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.TOOL_NAME}`,
          {
            vercelFunctionName: "ai.toolCall",
            vercelAttributes: {
              [vercelSemanticConvention]: "test-tool",
            },
            addedOpenInferenceAttributes: {
              [SemanticConventions.TOOL_NAME]: "test-tool",
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.TOOL,
            },
          },
        ]);
        break;
      case AISemanticConventions.TOOL_CALL_ARGS:
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.TOOL_PARAMETERS} with params as input.value for tool spans`,
            {
              vercelFunctionName: "ai.toolCall",
              vercelAttributes: {
                [vercelSemanticConvention]: JSON.stringify({ test1: "test-1" }),
              },
              addedOpenInferenceAttributes: {
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
            `${vercelSemanticConvention} to ${SemanticConventions.TOOL_PARAMETERS} with no input.value for non-tool spans`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: "test-args",
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.TOOL_PARAMETERS]: "test-args",
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
              },
            },
          ],
        );
        break;
      case AISemanticConventions.TOOL_CALL_RESULT:
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.OUTPUT_VALUE} with MIME type ${MimeType.TEXT}`,
            {
              vercelFunctionName: "ai.toolCall",
              vercelAttributes: {
                [vercelSemanticConvention]: "test-result",
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
                [SemanticConventions.OUTPUT_VALUE]: "test-result",
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.TOOL,
              },
            },
          ],
          [
            `${vercelSemanticConvention} to ${SemanticConventions.OUTPUT_VALUE} with MIME type ${MimeType.JSON}`,
            {
              vercelFunctionName: "ai.toolCall",
              vercelAttributes: {
                [vercelSemanticConvention]: JSON.stringify({ key: "value" }),
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.OUTPUT_VALUE]: JSON.stringify({
                  key: "value",
                }),
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.TOOL,
              },
            },
          ],
          [
            `${vercelSemanticConvention} to nothing if not a tool call`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: JSON.stringify({ key: "value" }),
              },
              addedOpenInferenceAttributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
              },
            },
          ],
        );
        break;
      default:
        assertUnreachable(vercelSemanticConvention);
    }
  });
  return testCases;
};

let traceProvider = new BasicTracerProvider();
let memoryExporter = new InMemorySpanExporter();
let processor:
  | OpenInferenceSimpleSpanProcessor
  | OpenInferenceBatchSpanProcessor;
function setupTraceProvider({
  Processor,
  spanFilters,
}: {
  Processor:
    | typeof OpenInferenceBatchSpanProcessor
    | typeof OpenInferenceSimpleSpanProcessor;
  spanFilters?: SpanFilter[];
}) {
  memoryExporter.reset();
  trace.disable();
  traceProvider = new BasicTracerProvider();
  memoryExporter = new InMemorySpanExporter();
  processor = new Processor({
    exporter: memoryExporter,
    spanFilters,
  });
  traceProvider.addSpanProcessor(processor);
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
      { vercelFunctionName, vercelAttributes, addedOpenInferenceAttributes },
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
      expect(spans[0].attributes).toStrictEqual({
        ...vercelAttributesWithOperationName,
        ...addedOpenInferenceAttributes,
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
  it("should export all spans if there are no filters", () => {
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
      spanFilters: [isOpenInferenceSpan],
    });
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("not ai");
    span.setAttribute("operation.name", "not ai stuff");
    span.end();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(0);
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
      { vercelFunctionName, vercelAttributes, addedOpenInferenceAttributes },
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
      expect(spans[0].attributes).toStrictEqual({
        ...vercelAttributesWithOperationName,
        ...addedOpenInferenceAttributes,
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
  it("should export all spans if there are no filters", async () => {
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
      spanFilters: [isOpenInferenceSpan],
    });
    const tracer = trace.getTracer("test-tracer");
    const span = tracer.startSpan("not ai");
    span.setAttribute("operation.name", "not ai stuff");
    span.end();
    await processor.forceFlush();
    const spans = memoryExporter.getFinishedSpans();
    expect(spans.length).toBe(0);
  });
});
