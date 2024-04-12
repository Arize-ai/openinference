import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

type LLMMessageToolCall = {
  [SemanticConventions.TOOL_CALL_FUNCTION_NAME]?: string;
  [SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]?: string;
};

export type LLMMessageToolCalls = {
  [SemanticConventions.MESSAGE_TOOL_CALLS]?: LLMMessageToolCall[];
};

export type LLMMessageFunctionCall = {
  [SemanticConventions.MESSAGE_FUNCTION_CALL_NAME]?: string;
  [SemanticConventions.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON]?: string;
};

export type LLMMessage = LLMMessageToolCalls &
  LLMMessageFunctionCall & {
    [SemanticConventions.MESSAGE_ROLE]?: string;
    [SemanticConventions.MESSAGE_CONTENT]?: string;
  };

export type LLMMessagesAttributes =
  | {
      [SemanticConventions.LLM_INPUT_MESSAGES]: LLMMessage[];
    }
  | {
      [SemanticConventions.LLM_OUTPUT_MESSAGES]: LLMMessage[];
    };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type GenericFunction = (...args: any[]) => any;

export type SafeFunction<T extends GenericFunction> = (
  ...args: Parameters<T>
) => ReturnType<T> | null;

export type RetrievalDocument = {
  [SemanticConventions.DOCUMENT_CONTENT]?: string;
  [SemanticConventions.DOCUMENT_METADATA]?: string;
};

export type LLMOpenInferenceAttributes = {
  [SemanticConventions.LLM_MODEL_NAME]?: string;
  [SemanticConventions.LLM_INVOCATION_PARAMETERS]?: string;
};
