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
  VercelSemanticConventions,
  VercelSemanticConventionsList,
} from "../src/VercelSemanticConventions";
import { assertUnreachable } from "../src/typeUtils";

const traceProvider = new BasicTracerProvider();

type TestName = string;
type VercelFunctionName = string;
type ProvidedVercelAttributes = Attributes;
type ExpectedOpenInferenceAttributes = Attributes;

type SpanProcessorTestCase = [
  TestName,
  VercelFunctionName,
  ProvidedVercelAttributes,
  ExpectedOpenInferenceAttributes,
];

const generateVercelAttributeTestCases = (): SpanProcessorTestCase[] => {
  const testCases: SpanProcessorTestCase[] = [];
  VercelSemanticConventionsList.map((vercelSemanticConvention) => {
    switch (vercelSemanticConvention) {
      case VercelSemanticConventions.MODEL_ID:
        testCases.push(
          [
            `${vercelSemanticConvention} (LLMs) to ${SemanticConventions.LLM_MODEL_NAME} for LLM`,
            "ai.generateText.doGenerate",
            { [vercelSemanticConvention]: "test-llm" },
            {
              [SemanticConventions.LLM_MODEL_NAME]: "test-llm",
            },
          ],
          [
            `${vercelSemanticConvention} (embeddings) to ${SemanticConventions.EMBEDDING_MODEL_NAME} for embeddings`,
            "ai.embed.doEmbed",
            { [vercelSemanticConvention]: "test-embedding-model" },
            {
              [SemanticConventions.EMBEDDING_MODEL_NAME]:
                "test-embedding-model",
            },
          ],
        );
        break;
      case VercelSemanticConventions.METADATA:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.METADATA}`,
          "ai.generateText.doGenerate",
          {
            [`${vercelSemanticConvention}.key1`]: "value1",
            [`${vercelSemanticConvention}.key2`]: "value2",
          },
          {
            [`${SemanticConventions.METADATA}.key1`]: "value1",
            [`${SemanticConventions.METADATA}.key2`]: "value2",
          },
        ]);
        break;
      case VercelSemanticConventions.SETTINGS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.LLM_INVOCATION_PARAMETERS}, accumulating and stringifying all settings`,
          "ai.generateText.doGenerate",
          {
            [`${vercelSemanticConvention}.key1`]: "value1",
            [`${vercelSemanticConvention}.key2`]: "value2",
          },
          {
            [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify({
              key1: "value1",
              key2: "value2",
            }),
          },
        ]);
        break;
      case VercelSemanticConventions.TOKEN_COUNT_COMPLETION:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.LLM_TOKEN_COUNT_COMPLETION}`,
          "ai.generateText.doGenerate",
          {
            [vercelSemanticConvention]: 10,
          },
          {
            [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 10,
          },
        ]);
        break;
      case VercelSemanticConventions.TOKEN_COUNT_PROMPT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.LLM_TOKEN_COUNT_PROMPT}`,
          "ai.generateText.doGenerate",
          {
            [vercelSemanticConvention]: 10,
          },
          {
            [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 10,
          },
        ]);
        break;
      case VercelSemanticConventions.RESULT_TEXT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.OUTPUT_VALUE} with MIME type ${MimeType.TEXT}`,
          "ai.generateText.doGenerate",
          {
            [vercelSemanticConvention]: "hello",
          },
          {
            [SemanticConventions.OUTPUT_VALUE]: "hello",
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
          },
        ]);
        break;
      case VercelSemanticConventions.RESULT_OBJECT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.OUTPUT_VALUE} with MIME type ${MimeType.JSON}`,
          "ai.generateObject.doGenerate",
          {
            [vercelSemanticConvention]: JSON.stringify({ key: "value" }),
          },
          {
            [SemanticConventions.OUTPUT_VALUE]: JSON.stringify({
              key: "value",
            }),
            [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
          },
        ]);
        break;
      case VercelSemanticConventions.RESULT_TOOL_CALLS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.MESSAGE_TOOL_CALLS} on ${SemanticConventions.LLM_OUTPUT_MESSAGES}`,
          "ai.toolCall",
          {
            [vercelSemanticConvention]: JSON.stringify([
              { toolName: "test-tool-1", args: { test1: "test-1" } },
              { toolName: "test-tool-2", args: { test2: "test-2" } },
            ]),
          },
          {
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
              "test-tool-1",
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.0.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
              JSON.stringify({ test1: "test-1" }),
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.1.${SemanticConventions.TOOL_CALL_FUNCTION_NAME}`]:
              "test-tool-2",
            [`${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_TOOL_CALLS}.1.${SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}`]:
              JSON.stringify({ test2: "test-2" }),
          },
        ]);
        break;
      case VercelSemanticConventions.PROMPT:
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.INPUT_VALUE} with MIME type ${MimeType.TEXT} for normal strings`,
            "ai.generateText.doGenerate",
            {
              [vercelSemanticConvention]: "hello",
            },
            {
              [SemanticConventions.INPUT_VALUE]: "hello",
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
            },
          ],
          [
            `${vercelSemanticConvention} to ${SemanticConventions.INPUT_VALUE} with MIME type ${MimeType.JSON} for JSON object strings`,
            "ai.generateText.doGenerate",
            {
              [vercelSemanticConvention]: JSON.stringify({}),
            },
            {
              [SemanticConventions.INPUT_VALUE]: "{}",
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
            },
          ],
        );
        break;
      case VercelSemanticConventions.PROMPT_MESSAGES:
        testCases.push(
          [
            `${vercelSemanticConvention} to ${SemanticConventions.LLM_INPUT_MESSAGES} with role and content for string content messages`,
            "ai.generateText.doGenerate",
            {
              [vercelSemanticConvention]: JSON.stringify([
                { role: "assistant", content: "hello" },
                { role: "user", content: "world" },
              ]),
            },
            {
              [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_ROLE}`]:
                "assistant",
              [`${SemanticConventions.LLM_INPUT_MESSAGES}.0.${SemanticConventions.MESSAGE_CONTENT}`]:
                "hello",
              [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_ROLE}`]:
                "user",
              [`${SemanticConventions.LLM_INPUT_MESSAGES}.1.${SemanticConventions.MESSAGE_CONTENT}`]:
                "world",
            },
          ],
          [
            `${vercelSemanticConvention} to ${SemanticConventions.LLM_INPUT_MESSAGES} with role and content for object content messages`,
            "ai.generateText.doGenerate",
            {
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
            {
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
          ],
        );
        break;
      case VercelSemanticConventions.EMBEDDING_TEXT:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_TEXT}`,
          "ai.embed.doEmbed",
          {
            [vercelSemanticConvention]: "hello",
          },
          {
            [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
              "hello",
          },
        ]);
        break;
      case VercelSemanticConventions.EMBEDDING_TEXTS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_TEXT} for multiple texts`,
          "ai.embedMany.doEmbed",
          {
            [vercelSemanticConvention]: ["hello", "world"],
          },
          {
            [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_TEXT}`]:
              "hello",
            [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_TEXT}`]:
              "world",
          },
        ]);
        break;
      case VercelSemanticConventions.EMBEDDING_VECTOR:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_VECTOR}`,
          "ai.embedMany.doEmbed",
          {
            [vercelSemanticConvention]: JSON.stringify([1, 2]),
          },
          {
            [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
              [1, 2],
          },
        ]);
        break;
      case VercelSemanticConventions.EMBEDDING_VECTORS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.EMBEDDING_VECTOR} for multiple vectors`,
          "ai.embedMany.doEmbed",
          {
            [vercelSemanticConvention]: ["[1, 2]", "[3, 4]"],
          },
          {
            [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.0.${SemanticConventions.EMBEDDING_VECTOR}`]:
              [1, 2],
            [`${SemanticConventions.EMBEDDING_EMBEDDINGS}.1.${SemanticConventions.EMBEDDING_VECTOR}`]:
              [3, 4],
          },
        ]);
        break;
      case VercelSemanticConventions.TOOL_CALL_NAME:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.TOOL_NAME}`,
          "ai.toolCall",
          {
            [vercelSemanticConvention]: "test-tool",
          },
          {
            [SemanticConventions.TOOL_NAME]: "test-tool",
          },
        ]);
        break;
      case VercelSemanticConventions.TOOL_CALL_ARGS:
        testCases.push([
          `${vercelSemanticConvention} to ${SemanticConventions.TOOL_PARAMETERS}`,
          "ai.toolCall",
          {
            [vercelSemanticConvention]: JSON.stringify({ test1: "test-1" }),
          },
          {
            [SemanticConventions.TOOL_PARAMETERS]: JSON.stringify({
              test1: "test-1",
            }),
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
    (_name, vercelFunctionName, providedAttributes, expectedAttributes) => {
      const tracer = trace.getTracer("test-tracer");
      const span = tracer.startSpan(vercelFunctionName);
      span.setAttribute("operation.name", vercelFunctionName);
      span.setAttributes(providedAttributes);
      span.end();
      const spans = memoryExporter.getFinishedSpans();
      expect(spans.length).toBe(1);
      expect(spans[0].attributes).toMatchObject({
        ...providedAttributes,
        ...expectedAttributes,
      });
    },
  );
});
