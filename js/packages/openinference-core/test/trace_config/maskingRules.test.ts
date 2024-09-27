import {
  DefaultTraceConfig,
  REDACTED_VALUE,
} from "@core/trace/trace_config/constants";
import { mask } from "@core/trace/trace_config/maskingRules";
import { TraceConfig, TraceConfigKey } from "@core/trace/trace_config/types";
import { assertUnreachable } from "@core/utils";
import { AttributeValue } from "@opentelemetry/api";

type Name = string;
type ExpectedValue = AttributeValue | undefined;
type AttributeKey = string;
type InitialValue = AttributeValue;

type MaskTestCases = [
  Name,
  TraceConfig,
  ExpectedValue,
  AttributeKey,
  InitialValue,
];
const generateMaskTestCases = (): MaskTestCases[] => {
  const testCases: MaskTestCases[] = [];
  Object.keys(DefaultTraceConfig).forEach((key) => {
    const configKey = key as TraceConfigKey;
    const ioValue = configKey.includes("Input") ? "input" : "output";
    switch (configKey) {
      case "hideInputMessages":
      case "hideOutputMessages":
      case "hideInputs":
      case "hideOutputs": {
        testCases.push([
          `should return undefined for "llm.${ioValue}_messages" when ${configKey} is set to true`,
          { ...DefaultTraceConfig, [configKey]: true },
          undefined,
          `llm.${ioValue}_messages.0.message.content`,
          "some message content",
        ]);
        if (configKey === "hideInputs" || configKey === "hideOutputs") {
          testCases.push(
            [
              `should return ${REDACTED_VALUE} for ${ioValue}.value when ${configKey} is set to true`,
              { ...DefaultTraceConfig, [configKey]: true },
              REDACTED_VALUE,
              `${ioValue}.value`,
              "some content",
            ],
            [
              `should return undefined for ${ioValue}.mime_type when ${configKey} is set to true`,
              { ...DefaultTraceConfig, [configKey]: true },
              undefined,
              `${ioValue}.mime_type`,
              "some content",
            ],
          );
        }
        break;
      }
      case "hideOutputText":
      case "hideInputText":
        testCases.push(
          [
            `should return ${REDACTED_VALUE} for "llm.${ioValue}_messages.0.message.content" when ${configKey} is set to true`,
            { ...DefaultTraceConfig, [configKey]: true },
            REDACTED_VALUE,
            `llm.${ioValue}_messages.0.message.content`,
            "some message content",
          ],
          [
            `should return ${REDACTED_VALUE} for "llm.${ioValue}_messages.0.message.contents.0.message_content.text" when ${configKey} is set to true`,
            { ...DefaultTraceConfig, [configKey]: true },
            REDACTED_VALUE,
            `llm.${ioValue}_messages.0.message.contents.0.message_content.text`,
            "some message content",
          ],
          [
            `should not change the value for "llm.${ioValue}_messages.0.message.contents.0.message_content.image" when ${configKey} is set to true`,
            { ...DefaultTraceConfig, [configKey]: true },
            "some image",
            `llm.${ioValue}_messages.0.message.contents.0.message_content.image`,
            "some image",
          ],
        );
        break;
      case "hideInputImages":
        testCases.push([
          `should return undefined for "llm.input_messages.0.message.contents.0.message_content.image" when hideInputImages is set to true`,
          { ...DefaultTraceConfig, hideInputImages: true },
          undefined,
          "llm.input_messages.0.message.contents.0.message_content.image",
          "some image",
        ]);
        break;
      case "hideEmbeddingVectors":
        testCases.push([
          `should return undefined for "embedding.embeddings.0.embedding.vector" when hideEmbeddingVectors is set to true`,
          { ...DefaultTraceConfig, hideEmbeddingVectors: true },
          undefined,
          "embedding.embeddings.0.embedding.vector",
          "some embedding vector",
        ]);
        break;
      case "base64ImageMaxLength":
        testCases.push([
          `should return ${REDACTED_VALUE} for "llm.input_messages.0.message.contents.0.message_content.image.url" when the base64 image is too long`,
          { ...DefaultTraceConfig, base64ImageMaxLength: 10 },
          REDACTED_VALUE,
          "llm.input_messages.0.message.contents.0.message_content.image.url",
          "data:image/base64,verylongbase64string",
        ]);
        break;
      default:
        assertUnreachable(configKey);
    }
  });
  return testCases;
};

describe("mask", () => {
  test.each(generateMaskTestCases())(
    "%s",
    (_, config, expected, key, initialValue) => {
      expect(mask({ config, key, value: initialValue })).toEqual(expected);
    },
  );
});
