import { SemanticConventions } from "@arizeai/openinference-semantic-conventions";

type LLMMessageToolCall = {
  [SemanticConventions.TOOL_CALL_FUNCTION_NAME]?: string;
  [SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON]?: string;
};
export type LLMMessage = {
  [SemanticConventions.MESSAGE_ROLE]?: string;
  [SemanticConventions.MESSAGE_CONTENT]?: string;
  [SemanticConventions.MESSAGE_FUNCTION_CALL_NAME]?: string;
  [SemanticConventions.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON]?: string;
  [SemanticConventions.MESSAGE_TOOL_CALLS]?: LLMMessageToolCall[];
};

export type LLMMessageFunctionCall = Pick<
  LLMMessage,
  "message.function_call_name" | "message.function_call_arguments_json"
>;

export type LLMMessageToolCalls = Pick<LLMMessage, "message.tool_calls">;

export type LLMMessagesAttributes =
  | {
      [SemanticConventions.LLM_INPUT_MESSAGES]: LLMMessage[];
    }
  | {
      [SemanticConventions.LLM_OUTPUT_MESSAGES]: LLMMessage[];
    };
