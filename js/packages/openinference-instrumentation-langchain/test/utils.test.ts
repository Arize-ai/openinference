import {
  MimeType,
  OpenInferenceSpanKind,
  RetrievalAttributePostfixes,
  SemanticAttributePrefixes,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";
import {
  safelyFlattenAttributes,
  safelyFormatFunctionCalls,
  safelyFormatIO,
  safelyFormatInputMessages,
  safelyFormatLLMParams,
  safelyFormatMetadata,
  safelyFormatOutputMessages,
  safelyFormatPromptTemplate,
  safelyFormatRetrievalDocuments,
  safelyFormatTokenCounts,
  safelyFormatToolCalls,
  safelyGetOpenInferenceSpanKindFromRunType,
  withSafety,
} from "../src/utils";
import { diag } from "@opentelemetry/api";
import { getLangchainMessage, getLangchainRun } from "./fixtures";
import { LLMMessage } from "../src/types";

describe("withSafety", () => {
  afterEach(() => {
    jest.clearAllMocks();
    jest.resetModules();
    jest.restoreAllMocks();
  });
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
    expect(safelyFormatIO({ io: undefined, ioType: "output" })).toEqual({});
  });

  it("should return the formatted the stringified IO when given an object with more than one key", () => {
    const inputResult = safelyFormatIO({ io: testInput, ioType: "input" });
    expect(inputResult).toEqual({
      [SemanticConventions.INPUT_VALUE]: JSON.stringify(testInput),
      [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
    });
    const outputResult = safelyFormatIO({ io: testOutput, ioType: "output" });
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
    const inputResult = safelyFormatIO({ io: simpleInput, ioType: "input" });
    expect(inputResult).toEqual({
      [SemanticConventions.INPUT_VALUE]: "input",
      [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
    });

    const outputResult = safelyFormatIO({ io: simpleOutput, ioType: "output" });
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

describe("formatRetrievalDocuments", () => {
  const runOutputDocuments = [{ pageContent: "doc1" }, { pageContent: "doc2" }];

  const expectedOpenInferenceRetrievalDocuments = {
    [`${SemanticAttributePrefixes.retrieval}.${RetrievalAttributePostfixes.documents}`]:
      [
        {
          [SemanticConventions.DOCUMENT_CONTENT]: "doc1",
        },
        {
          [SemanticConventions.DOCUMENT_CONTENT]: "doc2",
        },
      ],
  };
  it("should return null if run_type is not 'retriever'", () => {
    const result = safelyFormatRetrievalDocuments(getLangchainRun());
    expect(result).toBeNull();
  });

  it("should return null if outputs is not an object", () => {
    const result = safelyFormatRetrievalDocuments(
      getLangchainRun({ run_type: "retriever" }),
    );
    expect(result).toBeNull();
  });

  it("should return null if outputs.documents is not an array", () => {
    const result = safelyFormatRetrievalDocuments(
      getLangchainRun({
        run_type: "retriever",
        outputs: { documents: "not an array" },
      }),
    );
    expect(result).toBeNull();
  });

  it("should return an object with parsed documents if outputs.documents is an array", () => {
    const result = safelyFormatRetrievalDocuments(
      getLangchainRun({
        run_type: "retriever",
        outputs: { documents: runOutputDocuments },
      }),
    );
    expect(result).toEqual(expectedOpenInferenceRetrievalDocuments);
  });
  it("should filter out non object documents", () => {
    const result = safelyFormatRetrievalDocuments(
      getLangchainRun({
        run_type: "retriever",
        outputs: {
          documents: [...runOutputDocuments, "not an object"],
        },
      }),
    );
    expect(result).toEqual(expectedOpenInferenceRetrievalDocuments);
  });

  it("should add document metadata", () => {
    const result = safelyFormatRetrievalDocuments(
      getLangchainRun({
        run_type: "retriever",
        outputs: {
          documents: [{ pageContent: "doc1", metadata: { key: "value" } }],
        },
      }),
    );
    expect(result).toEqual({
      [`${SemanticAttributePrefixes.retrieval}.${RetrievalAttributePostfixes.documents}`]:
        [
          {
            [SemanticConventions.DOCUMENT_CONTENT]: "doc1",
            [SemanticConventions.DOCUMENT_METADATA]: JSON.stringify({
              key: "value",
            }),
          },
        ],
    });
  });
});
describe("formatLLMParams", () => {
  afterEach(() => {
    jest.clearAllMocks();
    jest.resetModules();
    jest.restoreAllMocks();
  });
  it("should return null if runExtra is not an object or runExtra.invocation_params is not an object", () => {
    expect(safelyFormatLLMParams(undefined)).toBeNull();
    expect(safelyFormatLLMParams({ test: "test" })).toBeNull();
    expect(safelyFormatLLMParams([])).toBeNull();
  });

  it("should return swallow errors stringifying invocation params, but still add model_name if possible", () => {
    const diagMock = jest.spyOn(diag, "error");
    const runExtra = getLangchainRun({
      extra: { invocation_params: { badKey: BigInt(1), model_name: "gpt-4" } },
    }).extra;
    const result = safelyFormatLLMParams(runExtra);
    expect(result).toStrictEqual({
      [SemanticConventions.LLM_INVOCATION_PARAMETERS]: undefined,
      [SemanticConventions.LLM_MODEL_NAME]: "gpt-4",
    });
    expect(diagMock).toHaveBeenCalledWith(
      "Failed to get attributes for span: TypeError: Do not know how to serialize a BigInt",
    );
  });

  it("should return the formatted LLMParameterAttributes object", () => {
    const runExtra = {
      invocation_params: {
        model_name: "test",
      },
    };
    const expectedParams = {
      [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify(
        runExtra.invocation_params,
      ),
      [SemanticConventions.LLM_MODEL_NAME]: "test",
    };
    const result = safelyFormatLLMParams(runExtra);
    expect(result).toEqual(expectedParams);
  });

  it("should use 'model' if 'model_name' is not a string", () => {
    const runExtra = {
      invocation_params: {
        model: "test",
      },
    };
    const expectedParams = {
      [SemanticConventions.LLM_INVOCATION_PARAMETERS]: JSON.stringify(
        runExtra.invocation_params,
      ),
      [SemanticConventions.LLM_MODEL_NAME]: "test",
    };
    const result = safelyFormatLLMParams(runExtra);
    expect(result).toEqual(expectedParams);
  });

  it("should return null if invocation_params is missing", () => {
    const runExtra = {};
    const result = safelyFormatLLMParams(runExtra);
    expect(result).toBeNull();
  });
});

describe("formatPromptTemplate", () => {
  it("should return null if run type is not 'prompt'", () => {
    const result = safelyFormatPromptTemplate(getLangchainRun());
    expect(result).toBeNull();
  });

  it("should return the run input as prompt template variables for a prompt run", () => {
    const promptRun = getLangchainRun({
      run_type: "prompt",
      inputs: { test: "test value" },
    });
    const result = safelyFormatPromptTemplate(promptRun);
    expect(result).toEqual({
      [SemanticConventions.PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(
        promptRun.inputs,
      ),
    });
  });

  it("should not return the template from the runs serialized field if it the path to the template has the wrong type", () => {
    const promptRun = getLangchainRun({
      run_type: "prompt",
      inputs: { test: "test value" },
      serialized: {
        kwargs: {
          messages: [
            {
              prompt: {
                template: "my template",
              },
            },
          ],
        },
      },
    });
    const result = safelyFormatPromptTemplate(promptRun);
    expect(result).toEqual({
      [SemanticConventions.PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(
        promptRun.inputs,
      ),
      [SemanticConventions.PROMPT_TEMPLATE_TEMPLATE]: "my template",
    });
  });

  it("should not return the template from the runs serialized field if it the path to the template has the wrong type", () => {
    const promptRun = getLangchainRun({
      run_type: "prompt",
      inputs: { test: "test value" },
      serialized: {
        kwargs: {
          messages: [
            {
              prompt: "my template",
            },
          ],
        },
      },
    });
    const result = safelyFormatPromptTemplate(promptRun);
    expect(result).toEqual({
      [SemanticConventions.PROMPT_TEMPLATE_VARIABLES]: JSON.stringify(
        promptRun.inputs,
      ),
    });
  });
});

describe("formatTokenCounts", () => {
  it("should return null if outputs is undefined", () => {
    const result = safelyFormatTokenCounts(undefined);
    expect(result).toBeNull();
  });

  it("should return null if llmOutput or tokenUsage is not an object", () => {
    const outputs = {
      llmOutput: null,
    };
    const result = safelyFormatTokenCounts(outputs);
    expect(result).toBeNull();
  });

  it("should return the token counts if llmOutput and tokenUsage are valid", () => {
    const outputs = {
      llmOutput: {
        tokenUsage: {
          completionTokens: 10,
          promptTokens: 20,
          totalTokens: 30,
        },
      },
    };
    const result = safelyFormatTokenCounts(outputs);
    expect(result).toEqual({
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 10,
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 20,
      [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: 30,
    });
  });

  it("should return null if llmOutput or tokenUsage are missing", () => {
    const outputs = {
      llmOutput: {},
    };
    const result = safelyFormatTokenCounts(outputs);
    expect(result).toBeNull();
  });

  it("should only add numeric token values", () => {
    const outputs = {
      llmOutput: {
        tokenUsage: {
          completionTokens: 10,
          promptTokens: 20,
          totalTokens: "wrong",
        },
      },
    };
    const result = safelyFormatTokenCounts(outputs);
    expect(result).toEqual({
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 10,
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 20,
    });
  });

  it("should add estimated token counts if actual usage is not present", () => {
    const outputs = {
      llmOutput: {
        estimatedTokenUsage: {
          completionTokens: 10,
          promptTokens: 20,
          totalTokens: 30,
        },
      },
    };
    const result = safelyFormatTokenCounts(outputs);
    expect(result).toEqual({
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]: 10,
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]: 20,
      [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: 30,
    });
  });
});

describe("formatFunctionCalls", () => {
  it("should return null if outputs is empty", () => {
    const result = safelyFormatFunctionCalls(undefined);
    expect(result).toBeNull();
  });

  it("should return null if the first output generation is not an object or does not have a message property", () => {
    const runWithInvalidGeneration = getLangchainRun({
      outputs: {
        generations: ["invalid generation"],
      },
    });
    const result = safelyFormatFunctionCalls(runWithInvalidGeneration.outputs);
    expect(result).toBeNull();

    const runWithInvalidMessage = getLangchainRun({
      outputs: {
        generations: [{ test: "invalid message" }],
      },
    });

    const resultWithNoMessage = safelyFormatFunctionCalls(
      runWithInvalidMessage.outputs,
    );
    expect(resultWithNoMessage).toBeNull();
  });

  it("should return null if additional_kwargs or function_call is not an object", () => {
    const runWithInvalidKwargs = getLangchainRun({
      outputs: {
        generations: [
          {
            message: {
              additional_kwargs: "invalid",
            },
          },
        ],
      },
    });
    const result = safelyFormatFunctionCalls(runWithInvalidKwargs.outputs);
    expect(result).toBeNull();

    const runWithInvalidFunctionCall = getLangchainRun({
      outputs: {
        generations: [
          {
            message: {
              additional_kwargs: {
                function_call: "invalid",
              },
            },
          },
        ],
      },
    });
    const resultWithInvalidFunctionCall = safelyFormatFunctionCalls(
      runWithInvalidFunctionCall.outputs,
    );

    expect(resultWithInvalidFunctionCall).toBeNull();
  });
});

it("should return the formatted function call if it is valid", () => {
  const validRun = getLangchainRun({
    outputs: {
      generations: [
        [
          {
            message: {
              additional_kwargs: {
                function_call: {
                  name: "test-function",
                  args: "my_arg: 'test'",
                },
              },
            },
          },
        ],
      ],
    },
  });
  const result = safelyFormatFunctionCalls(validRun.outputs);
  expect(result).toEqual({
    [SemanticConventions.LLM_FUNCTION_CALL]: JSON.stringify({
      name: "test-function",
      args: "my_arg: 'test'",
    }),
  });
});
describe("formatToolCalls", () => {
  it("should return null if the normalized run type is not 'tool'", () => {
    const run = getLangchainRun();
    const result = safelyFormatToolCalls(run);
    expect(result).toBeNull();
  });

  it("should get tool name from run name if not present in serialized field", () => {
    const run = getLangchainRun({ run_type: "tool", name: "test_tool" });
    const result = safelyFormatToolCalls(run);
    expect(result).toEqual({
      [SemanticConventions.TOOL_NAME]: "test_tool",
    });
  });

  it("should return the tool attributes with the serialized name and description if available", () => {
    const run = getLangchainRun({
      run_type: "tool",
      name: "Test Run",
      serialized: {
        name: "my_tools_name",
        description: "my tools description",
      },
    });
    const result = safelyFormatToolCalls(run);
    expect(result).toEqual({
      [SemanticConventions.TOOL_NAME]: "my_tools_name",
      [SemanticConventions.TOOL_DESCRIPTION]: "my tools description",
    });
  });

  it("should not return name or description if they are not strings", () => {
    const run = getLangchainRun({
      run_type: "tool",
      name: "test_tool",
      serialized: {
        name: 1,
        description: 1,
      },
    });
    const result = safelyFormatToolCalls(run);
    expect(result).toEqual({
      [SemanticConventions.TOOL_NAME]: "test_tool",
    });
  });
});

describe("formatMetadata", () => {
  it("should return null if run.extra or run.extra.metadata is not an object", () => {
    const run1 = getLangchainRun();
    const run2 = getLangchainRun({ extra: { metadata: null } });
    const run3 = getLangchainRun({ extra: { metadata: "invalid" } });

    expect(safelyFormatMetadata(run1)).toBeNull();
    expect(safelyFormatMetadata(run2)).toBeNull();
    expect(safelyFormatMetadata(run3)).toBeNull();
  });

  it("should return the formatted metadata if run.extra.metadata is an object", () => {
    const run = getLangchainRun({ extra: { metadata: { key: "value" } } });
    const expected = { metadata: JSON.stringify({ key: "value" }) };

    expect(safelyFormatMetadata(run)).toEqual(expected);
  });
});
