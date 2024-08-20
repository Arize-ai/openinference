import { Attributes, trace } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  InMemorySpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { VercelSDKFunctionNameToSpanKindMap } from "../src/constants";
import { OpenInferenceSpanProcessor } from "../src";
import {
  MimeType,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  AISemanticConventions,
  AISemanticConventionsList,
} from "../src/AISemanticConventions";
import { assertUnreachable } from "../src/typeUtils";

const traceProvider = new BasicTracerProvider();

type SpanProcessorTestCase = [
  string,
  {
    vercelFunctionName: string;
    vercelAttributes: Attributes;
    expectedAttributes: Attributes;
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
              expectedAttributes: {
                [SemanticConventions.LLM_MODEL_NAME]: "test-llm",
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
              expectedAttributes: {
                [SemanticConventions.EMBEDDING_MODEL_NAME]:
                  "test-embedding-model",
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
            expectedAttributes: {
              [`${SemanticConventions.METADATA}.key1`]: "value1",
              [`${SemanticConventions.METADATA}.key2`]: "value2",
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
            expectedAttributes: {
              [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify({
                key1: "value1",
                key2: "value2",
              }),
            },
          },
        ]);
        break;
      case AISemanticConventions.TOKEN_COUNT_COMPLETION:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.LLM_TOKEN_COUNT_COMPLETION}`,
          {
            vercelFunctionName: "ai.generateText.doGenerate",
            vercelAttributes: {
              [vercelSemanticConvention]: 10,
            },
            expectedAttributes: {
              [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 10,
            },
          },
        ]);
        break;
      case AISemanticConventions.TOKEN_COUNT_PROMPT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}`,
          {
            vercelFunctionName: "ai.generateText.doGenerate",
            vercelAttributes: {
              [vercelSemanticConvention]: 10,
            },
            expectedAttributes: {
              [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 10,
            },
          },
        ]);
        break;
      case AISemanticConventions.RESULT_TEXT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.OUTPUT_VALUE} with MIME type ${MimeType.TEXT}`,
          {
            vercelFunctionName: "ai.generateText.doGenerate",
            vercelAttributes: {
              [vercelSemanticConvention]: "hello",
            },
            expectedAttributes: {
              [SemanticConventions.OUTPUT_VALUE]: "hello",
              [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
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
            expectedAttributes: {
              [SemanticConventions.OUTPUT_VALUE]: JSON.stringify({
                key: "value",
              }),
              [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
            },
          },
        ]);
        break;
      case AISemanticConventions.RESULT_TOOL_CALLS:
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
            expectedAttributes: {
              [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
                "test-tool-1",
              [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
                JSON.stringify({ test1: "test-1" }),
              [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.1.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
                "test-tool-2",
              [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.1.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
                JSON.stringify({ test2: "test-2" }),
            },
          },
        ]);
        break;
      case AISemanticConventions.PROMPT:
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.INPUT_VALUE} with MIME type ${MimeType.TEXT} for normal strings`,
            {
              vercelFunctionName: "ai.generateText.doGenerate",
              vercelAttributes: {
                [vercelSemanticConvention]: "hello",
              },
              expectedAttributes: {
                [SemanticConventions.INPUT_VALUE]: "hello",
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
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
              expectedAttributes: {
                [SemanticConventions.INPUT_VALUE]: "{}",
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
              },
            },
          ],
        );
        break;
      case AISemanticConventions.PROMPT_MESSAGES:
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
              expectedAttributes: {
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
                  "assistant",
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
                  "hello",
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]:
                  "user",
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
                  "world",
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
              expectedAttributes: {
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
                  "assistant",
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
                  "text",
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.0.${SemanticConventions.MESSAGE_CONTENT_TEXT}`]:
                  "hello",
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.1.${SemanticConventions.MESSAGE_CONTENT_TYPE}`]:
                  "image",
                [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENTS}.1.${SemanticConventions.MESSAGE_CONTENT_IMAGE}`]:
                  "image.com",
              },
            },
          ],
        );
        break;
      case AISemanticConventions.EMBEDDING_TEXT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_TEXT}`,
          {
            vercelFunctionName: "ai.embed.doEmbed",
            vercelAttributes: {
              [vercelSemanticConvention]: "hello",
            },
            expectedAttributes: {
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
                "hello",
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
            expectedAttributes: {
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
                "hello",
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_TEXT}`]:
                "world",
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
            expectedAttributes: {
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
                [1, 2],
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
            expectedAttributes: {
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
                [1, 2],
              [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_VECTOR}`]:
                [3, 4],
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
            expectedAttributes: {
              [SemanticConventions.TOOL_NAME]: "test-tool",
            },
          },
        ]);
        break;
      case AISemanticConventions.TOOL_CALL_ARGS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.TOOL_PARAMETERS}`,
          {
            vercelFunctionName: "ai.toolCall",
            vercelAttributes: {
              [vercelSemanticConvention]: JSON.stringify({ test1: "test-1" }),
            },
            expectedAttributes: {
              [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify({
                test1: "test-1",
              }),
            },
          },
        ]);
        break;
      default:
        assertUnreachable(vercelSemanticConvention);
    }
  });
  return testCases;
};

describe("OpenInferenceSpanProcessor", () => {
  const memoryExporter = new InMemorySpanExporter();
  traceProvider.addSpanProcessor(new OpenInferenceSpanProcessor());
  traceProvider.addSpanProcessor(new SimpleSpanProcessor(memoryExporter));
  trace.setGlobalTracerProvider(traceProvider);

  beforeEach(() => {
    memoryExporter.reset();
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
    (_name, { vercelFunctionName, vercelAttributes, expectedAttributes }) => {
      const tracer = trace.getTracer("test-tracer");
      const span = tracer.startSpan(vercelFunctionName);
      span.setAttribute("operation.name", vercelFunctionName);
      span.setAttributes(vercelAttributes);
      span.end();
      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].attributes).toMatchObject({
        ...vercelAttributes,
        ...expectedAttributes,
      });
    },
  );
});
