import type { AttributeValue } from "@opentelemetry/api";
import { describe, expect, test } from "vitest";

import { DefaultTraceConfig, REDACTED_VALUE } from "../../src/trace/trace-config/constants";
import { mask } from "../../src/trace/trace-config/maskingRules";
import type { TraceConfig, TraceConfigKey } from "../../src/trace/trace-config/types";
import { assertUnreachable } from "../../src/utils";

type Name = string;
type ExpectedValue = AttributeValue | undefined;
type AttributeKey = string;
type InitialValue = AttributeValue;

type MaskTestCases = [Name, TraceConfig, ExpectedValue, AttributeKey, InitialValue];
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
      case "hidePrompts":
        testCases.push([
          `should return ${REDACTED_VALUE} for "llm.prompts" when hidePrompts is set to true`,
          { ...DefaultTraceConfig, hidePrompts: true },
          REDACTED_VALUE,
          "llm.prompts",
          "some prompt",
        ]);
        break;
      case "hideLLMTools":
        testCases.push(
          [
            `should return undefined for "llm.tools.0.tool.json_schema" when hideLLMTools is set to true`,
            { ...DefaultTraceConfig, hideLLMTools: true },
            undefined,
            "llm.tools.0.tool.json_schema",
            '{"type":"function","function":{"name":"get_weather"}}',
          ],
          [
            `should return undefined for "llm.tools.0.tool.json_schema" when hideInputs is set to true`,
            { ...DefaultTraceConfig, hideInputs: true },
            undefined,
            "llm.tools.0.tool.json_schema",
            '{"type":"function","function":{"name":"get_weather"}}',
          ],
        );
        break;
      default:
        assertUnreachable(configKey);
    }
  });
  return testCases;
};

describe("mask", () => {
  test.each(generateMaskTestCases())("%s", (_, config, expected, key, initialValue) => {
    expect(mask({ config, key, value: initialValue })).toEqual(expected);
  });
});

describe("mask reasoning content fields", () => {
  // Opaque vendor-issued echo tokens (reasoning signature/data/encrypted_content
  // and tool_call.reasoning_signature) are not user-visible text. They should
  // pass through hideInputText/hideOutputText (which target message_content.text
  // only) but be removed by hideInputMessages/hideOutputMessages, since the
  // entire message tree is suppressed in that case.
  const opaqueInputKeys = [
    "llm.input_messages.0.message.contents.0.message_content.signature",
    "llm.input_messages.0.message.contents.0.message_content.data",
    "llm.input_messages.0.message.contents.0.message_content.encrypted_content",
    "llm.input_messages.0.message.tool_calls.0.tool_call.reasoning_signature",
  ];
  const opaqueOutputKeys = [
    "llm.output_messages.0.message.contents.0.message_content.signature",
    "llm.output_messages.0.message.contents.0.message_content.data",
    "llm.output_messages.0.message.contents.0.message_content.encrypted_content",
    "llm.output_messages.0.message.tool_calls.0.tool_call.reasoning_signature",
  ];

  test.each(opaqueInputKeys)(
    "%s is preserved when hideInputText is true (opaque echo token, not text)",
    (key) => {
      expect(
        mask({ config: { ...DefaultTraceConfig, hideInputText: true }, key, value: "token" }),
      ).toBe("token");
    },
  );

  test.each(opaqueOutputKeys)(
    "%s is preserved when hideOutputText is true (opaque echo token, not text)",
    (key) => {
      expect(
        mask({ config: { ...DefaultTraceConfig, hideOutputText: true }, key, value: "token" }),
      ).toBe("token");
    },
  );

  test("reasoning text is redacted by hideInputText (it is emitted as message_content.text)", () => {
    expect(
      mask({
        config: { ...DefaultTraceConfig, hideInputText: true },
        key: "llm.input_messages.0.message.contents.0.message_content.text",
        value: "let me think...",
      }),
    ).toBe(REDACTED_VALUE);
  });

  test("reasoning text is redacted by hideOutputText (it is emitted as message_content.text)", () => {
    expect(
      mask({
        config: { ...DefaultTraceConfig, hideOutputText: true },
        key: "llm.output_messages.0.message.contents.0.message_content.text",
        value: "let me think...",
      }),
    ).toBe(REDACTED_VALUE);
  });

  test.each(opaqueInputKeys)("%s is dropped when hideInputMessages is true", (key) => {
    expect(
      mask({ config: { ...DefaultTraceConfig, hideInputMessages: true }, key, value: "token" }),
    ).toBeUndefined();
  });

  test.each(opaqueOutputKeys)("%s is dropped when hideOutputMessages is true", (key) => {
    expect(
      mask({ config: { ...DefaultTraceConfig, hideOutputMessages: true }, key, value: "token" }),
    ).toBeUndefined();
  });
});
