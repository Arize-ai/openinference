import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  safelyFlattenAttributes,
  safelyFormatIO,
  safelyFormatInputMessages,
  safelyFormatOutputMessages,
  safelyGetOpenInferenceSpanKindFromRunType,
  withSafety,
} from "../src/utils";
import { diag } from "@opentelemetry/api";
import { getLangchainMessage } from "./fixtures";
import { LLMMessage } from "../src/types";

describe("withSafety", () => {
  it("should return a function", () => {
    const safeFunction = withSafety(() => {});
    expect(typeof safeFunction).toBe("function");
  });

  it("should execute the provided function without errors", () => {
    const mockFn = jest.fn();
    const safeFunction = withSafety(mockFn);
    safeFunction();
    expect(mockFn).toHaveBeenCalled();
  });

  it("should return null and log an error when the provided function throws an error", () => {
    const error = new Error("Test error");
    const mockFn = jest.fn((_a: number) => {
      throw error;
    });
    const diagMock = jest.spyOn(diag, "error");
    const safeFunction = withSafety(mockFn);
    const result = safeFunction(1);
    expect(result).toBeNull();
    expect(mockFn).toHaveBeenCalledWith(1);
    expect(diagMock).toHaveBeenCalledWith(
      `Failed to get attributes for span: ${error}`,
    );
  });
});

describe("safelyFlattenAttributes", () => {
  const testAttributes = {
    input: {
      value: "test",
      mime_type: "application/json",
    },
    llm: {
      token_count: {
        total: 10,
      },
    },
    test: "test value",
  };

  const expectedFlattenedAttributes = {
    "input.value": "test",
    "input.mime_type": "application/json",
    "llm.token_count.total": 10,
    test: "test value",
  };
  it("should flatten attributes with nested objects", () => {
    const result = safelyFlattenAttributes(testAttributes);
    expect(result).toEqual(expectedFlattenedAttributes);
  });

  it("should flatten attributes with nested arrays", () => {
    const result = safelyFlattenAttributes({
      ...testAttributes,
      list: ["1", "2"],
      listOfObjects: [{ key: "value" }, { key: "value2" }],
    });
    expect(result).toEqual({
      ...expectedFlattenedAttributes,
      "list.0": "1",
      "list.1": "2",
      "listOfObjects.0.key": "value",
      "listOfObjects.1.key": "value2",
    });
  });
  it("should ignore null and undefined values", () => {
    const result = safelyFlattenAttributes({
      ...testAttributes,
      nullValue: null,
      undefinedValue: undefined,
    });
    expect(result).toEqual(expectedFlattenedAttributes);
  });
});

describe("getOpenInferenceSpanKindFromRunType", () => {
  it("should return OpenInferenceSpanKind.AGENT when runType includes 'AGENT'", () => {
    const runType = "agent";
    const result = safelyGetOpenInferenceSpanKindFromRunType(runType);
    expect(result).toBe(OpenInferenceSpanKind.AGENT);
  });

  it("should return the corresponding OpenInferenceSpanKind value when runType is a valid key", () => {
    const runType = "LLM";
    const result = safelyGetOpenInferenceSpanKindFromRunType(runType);
    expect(result).toBe(OpenInferenceSpanKind.LLM);
  });

  it("should be case-insensitive", () => {
    const runType = "reTrIeVeR";
    const result = safelyGetOpenInferenceSpanKindFromRunType(runType);
    expect(result).toBe(OpenInferenceSpanKind.RETRIEVER);
  });

  it("should return 'UNKNOWN' when runType is not recognized", () => {
    const runType = "test";
    const result = safelyGetOpenInferenceSpanKindFromRunType(runType);
    expect(result).toBe("UNKNOWN");
  });
});
describe("formatIO", () => {
  const testInput = {
    value: "test input",
    metadata: "test input metadata",
  };
  const testOutput = {
    value: "test input",
    metadata: "test input metadata",
  };

  it("should return an empty object when output is undefined", () => {
    expect(safelyFormatIO({ io: undefined, type: "output" })).toEqual({});
  });

  it("should return the formatted the stringified IO when given an object with more than one key", () => {
    const inputResult = safelyFormatIO({ io: testInput, type: "input" });
    expect(inputResult).toEqual({
      [SemanticConventions.INPUT_VALUE]: JSON.stringify(testInput),
      [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
    });
    const outputResult = safelyFormatIO({ io: testOutput, type: "output" });
    expect(outputResult).toEqual({
      [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(testOutput),
      [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
    });
  });
  it("should return the io value as TEXT type when there is only one value in the io", () => {
    const simpleInput = {
      value: "input",
    };
    const simpleOutput = {
      value: "output",
    };
    const inputResult = safelyFormatIO({ io: simpleInput, type: "input" });
    expect(inputResult).toEqual({
      [SemanticConventions.INPUT_VALUE]: "input",
      [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
    });

    const outputResult = safelyFormatIO({ io: simpleOutput, type: "output" });
    expect(outputResult).toEqual({
      [SemanticConventions.OUTPUT_VALUE]: "output",
      [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
    });
  });
});

describe("formatMessages", () => {
  const testMessages = [
    [
      getLangchainMessage({
        lc_id: ["test ignore", "system"],
        lc_kwargs: { content: "system message" },
      }),
      getLangchainMessage(),
    ],
  ];
  const expectedMessages: LLMMessage[] = [
    {
      "message.content": "system message",
      "message.role": "system",
    },
    {
      "message.content": "hello, this is a test",
      "message.role": "user",
    },
  ];
  describe("formatInputMessages", () => {
    it("should return null if input messages is an empty array", () => {
      const result = safelyFormatInputMessages({
        messages: [],
      });
      expect(result).toBeNull();
    });

    it("should return null if the first set of messages is not an array", () => {
      const result = safelyFormatInputMessages({
        messages: [{}],
      });
      expect(result).toBeNull();
    });

    it("should parse and return the input messages if they are valid", () => {
      const result = safelyFormatInputMessages({
        messages: testMessages,
      });
      expect(result).toEqual({
        [SemanticConventions.LLM_INPUT_MESSAGES]: expectedMessages,
      });
    });

    it("should ignore non-object messages and return the valid ones", () => {
      const result = safelyFormatInputMessages({
        messages: [[...testMessages[0], "invalid message"]],
      });
      expect(result).toEqual({
        [SemanticConventions.LLM_INPUT_MESSAGES]: expectedMessages,
      });
    });

    it("should parse add tool call information", () => {
      const result = safelyFormatInputMessages({
        messages: [
          [
            ...testMessages[0],
            getLangchainMessage({
              lc_kwargs: {
                additional_kwargs: {
                  tool_calls: [
                    {
                      function: {
                        name: "test-tool",
                        arguments: "my_arg: 'test'",
                      },
                    },
                    {
                      function: {
                        name: "test-tool-2",
                        arguments: "my_arg_2: 1",
                      },
                    },
                  ],
                },
              },
            }),
          ],
        ],
      });
      expect(result).toEqual({
        [SemanticConventions.LLM_INPUT_MESSAGES]: [
          ...expectedMessages,
          {
            "message.role": "user",
            "message.tool_calls": [
              {
                "tool_call.function.arguments": "my_arg: 'test'",
                "tool_call.function.name": "test-tool",
              },
              {
                "tool_call.function.arguments": "my_arg_2: 1",
                "tool_call.function.name": "test-tool-2",
              },
            ],
          },
        ],
      });
    });
    it("should parse add function call information", () => {
      const result = safelyFormatInputMessages({
        messages: [
          [
            ...testMessages[0],
            getLangchainMessage({
              lc_kwargs: {
                additional_kwargs: {
                  function_call: {
                    name: "test-function",
                    args: "my_arg: 'test'",
                  },
                },
              },
            }),
          ],
        ],
      });
      expect(result).toEqual({
        [SemanticConventions.LLM_INPUT_MESSAGES]: [
          ...expectedMessages,
          {
            "message.role": "user",
            "message.function_call_arguments_json": "my_arg: 'test'",
            "message.function_call_name": "test-function",
          },
        ],
      });
    });
  });

  describe("formatOutputMessages", () => {
    const testOutputMessages = [
      testMessages[0].map((message) => ({ message })),
    ];
    it("should return null if generations is an empty array", () => {
      const result = safelyFormatOutputMessages({
        generations: [],
      });
      expect(result).toBeNull();
    });

    it("should return null if the first set of generations is not an array", () => {
      const result = safelyFormatOutputMessages({
        generations: [{}],
      });
      expect(result).toBeNull();
    });

    it("should parse and return the generations if they are valid", () => {
      const result = safelyFormatOutputMessages({
        generations: testOutputMessages,
      });
      expect(result).toEqual({
        [SemanticConventions.LLM_OUTPUT_MESSAGES]: expectedMessages,
      });
    });

    it("should ignore non-object messages and return the valid ones", () => {
      const result = safelyFormatOutputMessages({
        generations: [
          [
            ...testOutputMessages[0],
            "invalid message",
            { notGeneration: getLangchainMessage() },
          ],
        ],
      });
      expect(result).toEqual({
        [SemanticConventions.LLM_OUTPUT_MESSAGES]: expectedMessages,
      });
    });
  });
});
