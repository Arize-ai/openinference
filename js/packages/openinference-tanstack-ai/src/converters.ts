import {
  convertSchemaToJsonSchema,
  type ChatMiddlewareConfig,
  type ModelMessage,
  type StreamChunk,
  type Tool as TanStackTool,
  type ToolCall as TanStackToolCall,
} from "@tanstack/ai";

import type {
  Message as OpenInferenceMessage,
  OISpan,
  Tool as OpenInferenceTool,
  ToolCall as OpenInferenceToolCall,
} from "@arizeai/openinference-core";
import { getLLMAttributes, safelyJSONStringify } from "@arizeai/openinference-core";
import { MimeType } from "@arizeai/openinference-semantic-conventions";

import type { CurrentLLMSpanState, UsageInfo } from "./types";

/**
 * Applies prompt, completion, and total token counts to a span.
 */
export function setUsageAttributes(span: OISpan, usage: UsageInfo) {
  span.setAttributes(
    getLLMAttributes({
      tokenCount: {
        prompt: usage.promptTokens,
        completion: usage.completionTokens,
        total: usage.totalTokens,
      },
    }),
  );
}

/**
 * Converts TanStack model messages into the message shape expected by the
 * OpenInference core helpers.
 */
export function toOpenInferenceMessage(message: ModelMessage): OpenInferenceMessage {
  const result: OpenInferenceMessage = {
    role: message.role,
  };

  if (typeof message.content === "string") {
    result.content = message.content;
  }

  if (Array.isArray(message.toolCalls) && message.toolCalls.length > 0) {
    result.toolCalls = message.toolCalls.map(toOpenInferenceToolCallFromTanStack);
  }

  if (message.toolCallId != null) {
    result.toolCallId = message.toolCallId;
  }

  return result;
}

/**
 * Prepends system prompts so each LLM span captures the full effective prompt.
 */
export function getInputMessages(config: ChatMiddlewareConfig): OpenInferenceMessage[] {
  const systemMessages: OpenInferenceMessage[] = config.systemPrompts.map((prompt) => ({
    role: "system",
    content: prompt,
  }));
  return [...systemMessages, ...config.messages.map(toOpenInferenceMessage)];
}

/**
 * Converts TanStack tool call metadata into the OpenInference tool-call shape.
 */
export function toOpenInferenceToolCallFromTanStack(
  toolCall: TanStackToolCall,
): OpenInferenceToolCall {
  return {
    id: toolCall.id,
    function: {
      name: toolCall.function.name,
      arguments: toolCall.function.arguments,
    },
  };
}

/**
 * Captures tool definitions on LLM spans.
 */
export function toOpenInferenceTools(tools: TanStackTool[]): OpenInferenceTool[] {
  return tools.flatMap((tool) => {
    try {
      const inputSchema = tool.inputSchema;
      if (inputSchema == null) {
        return [];
      }
      return [
        {
          jsonSchema: {
            type: "function",
            function: {
              name: tool.name,
              description: tool.description,
              parameters: convertSchemaToJsonSchema(inputSchema),
            },
          },
        },
      ];
    } catch {
      return [];
    }
  });
}

/**
 * Collects the model-facing invocation parameters for the current model turn.
 */
export function getInvocationParameters(ctx: {
  model: string;
  config: ChatMiddlewareConfig;
}): Record<string, unknown> {
  return {
    model: ctx.model,
    temperature: ctx.config.temperature,
    topP: ctx.config.topP,
    maxTokens: ctx.config.maxTokens,
    metadata: ctx.config.metadata,
    modelOptions: ctx.config.modelOptions,
  };
}

/**
 * Builds the raw JSON payload stored in `input.value` for each LLM span.
 */
export function getLLMInputValue(options: {
  inputMessages: OpenInferenceMessage[];
  tools: OpenInferenceTool[];
  invocationParameters: Record<string, unknown>;
}): string | undefined {
  return (
    safelyJSONStringify({
      messages: options.inputMessages,
      tools: options.tools.map((tool) => tool.jsonSchema),
      invocationParameters: options.invocationParameters,
    }) ?? undefined
  );
}

/**
 * Derives the final assistant output for an LLM span from streamed text and
 * streamed tool-call events.
 */
export function getLLMOutput(options: {
  outputText: string;
  outputToolCalls: OpenInferenceToolCall[];
}): { value: string; mimeType: MimeType; outputMessages: OpenInferenceMessage[] } | undefined {
  if (options.outputText.length === 0 && options.outputToolCalls.length === 0) {
    return undefined;
  }

  const outputMessage: OpenInferenceMessage = {
    role: "assistant",
  };

  if (options.outputText.length > 0) {
    outputMessage.content = options.outputText;
  }

  if (options.outputToolCalls.length > 0) {
    outputMessage.toolCalls = options.outputToolCalls;
  }

  if (options.outputToolCalls.length > 0) {
    return {
      value:
        safelyJSONStringify({
          content: options.outputText.length > 0 ? options.outputText : undefined,
          tool_calls: options.outputToolCalls,
        }) ?? "{}",
      mimeType: MimeType.JSON,
      outputMessages: [outputMessage],
    };
  }

  return {
    value: options.outputText,
    mimeType: MimeType.TEXT,
    outputMessages: [outputMessage],
  };
}

/**
 * Starts tracking a tool call emitted by the model so it can be reflected in
 * the LLM span output.
 */
export function initializeToolCall(
  state: CurrentLLMSpanState,
  chunk: Extract<StreamChunk, { type: "TOOL_CALL_START" }>,
) {
  state.outputToolCalls.set(chunk.toolCallId, {
    id: chunk.toolCallId,
    function: {
      name: chunk.toolName,
      arguments: "",
    },
  });
}

/**
 * Appends or replaces streamed tool-call arguments as they arrive.
 */
export function updateToolCallArguments(
  state: CurrentLLMSpanState,
  chunk: Extract<StreamChunk, { type: "TOOL_CALL_ARGS" }>,
) {
  const existingToolCall = state.outputToolCalls.get(chunk.toolCallId);
  if (existingToolCall == null) {
    return;
  }
  existingToolCall.function = {
    ...existingToolCall.function,
    arguments: chunk.args ?? `${existingToolCall.function?.arguments ?? ""}${chunk.delta}`,
  };
}

/**
 * Finalizes the captured tool-call payload for the current LLM span.
 */
export function completeToolCall(
  state: CurrentLLMSpanState,
  chunk: Extract<StreamChunk, { type: "TOOL_CALL_END" }>,
) {
  const existingToolCall = state.outputToolCalls.get(chunk.toolCallId);
  if (existingToolCall == null) {
    return;
  }
  const toolArguments = chunk.input != null ? safelyJSONStringify(chunk.input) : undefined;
  existingToolCall.function = {
    ...existingToolCall.function,
    name: chunk.toolName,
    arguments: toolArguments ?? existingToolCall.function?.arguments,
  };
}
