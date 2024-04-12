import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";
import {
  PROMPT_TEMPLATE_TEMPLATE,
  PROMPT_TEMPLATE_VARIABLES,
  TOOL_DESCRIPTION,
  TOOL_NAME,
} from "./constants";

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

export type LLMParameterAttributes = {
  [SemanticConventions.LLM_MODEL_NAME]?: string;
  [SemanticConventions.LLM_INVOCATION_PARAMETERS]?: string;
};

export type PromptTemplateAttributes = {
  [PROMPT_TEMPLATE_TEMPLATE]?: string;
  [PROMPT_TEMPLATE_VARIABLES]?: string;
};
export type TokenCountAttributes = {
  [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]?: number;
  [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]?: number;
  [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]?: number;
};

export type ToolAttributes = {
  [TOOL_NAME]?: string;
  [TOOL_DESCRIPTION]?: string;
};
