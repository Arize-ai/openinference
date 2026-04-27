import type { Tracer } from "@opentelemetry/api";
import { context, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  convertSchemaToJsonSchema,
  type ChatMiddleware,
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
  TraceConfigOptions,
} from "@arizeai/openinference-core";
import {
  getInputAttributes,
  getLLMAttributes,
  getMetadataAttributes,
  getOutputAttributes,
  getToolAttributes,
  OITracer,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { VERSION } from "./version";

const INSTRUMENTATION_NAME = "@arizeai/openinference-tanstack-ai";
const AGENT_SPAN_NAME = "TanStack AI chat";
const TOOL_SPAN_PREFIX = "TanStack AI tool: ";
const LLM_SPAN_PREFIX = "TanStack AI LLM ";

type UsageInfo = NonNullable<Extract<StreamChunk, { type: "RUN_FINISHED" }>["usage"]>;

type RequestState = {
  chatSpan: OISpan;
  toolSpans: Map<string, OISpan>;
  currentLLMSpan?: {
    span: OISpan;
    outputText: string;
    outputToolCalls: Map<string, OpenInferenceToolCall>;
  };
};

/**
 * Configuration for the TanStack AI middleware.
 *
 * `tracer` lets consumers opt into a specific OTel tracer while still using
 * the shared OpenInference span shaping in this package.
 */
export type OpenInferenceTanStackAIMiddlewareOptions = {
  tracer?: Tracer;
  traceConfig?: TraceConfigOptions;
};

/**
 * Extracts a human-readable tool description when TanStack provides one.
 */
function getToolDescription(tool: TanStackTool | undefined): string | undefined {
  if (tool == null || typeof tool !== "object") {
    return undefined;
  }
  const description = Reflect.get(tool, "description");
  return typeof description === "string" ? description : undefined;
}

/**
 * Normalizes arbitrary tool arguments into an object for helper utilities.
 */
function toRecord(value: unknown): Record<string, unknown> {
  if (value != null && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return { value };
}

/**
 * Applies prompt, completion, and total token counts to a span.
 */
function setUsageAttributes(span: OISpan, usage: UsageInfo) {
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
 * Marks unfinished tool spans as failed before ending them.
 */
function endToolSpansWithError(toolSpans: Map<string, OISpan>, message: string) {
  toolSpans.forEach((span) => {
    span.setStatus({ code: SpanStatusCode.ERROR, message });
    span.end();
  });
}

/**
 * Ends a span if it was created.
 */
function finalizeSpan(span: OISpan | undefined) {
  span?.end();
}

/**
 * Converts TanStack model messages into the message shape expected by the
 * OpenInference core helpers.
 */
function toOpenInferenceMessage(message: ModelMessage): OpenInferenceMessage {
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
function getInputMessages(config: ChatMiddlewareConfig): OpenInferenceMessage[] {
  const systemMessages: OpenInferenceMessage[] = config.systemPrompts.map((prompt) => ({
    role: "system",
    content: prompt,
  }));
  return [...systemMessages, ...config.messages.map(toOpenInferenceMessage)];
}

/**
 * Converts TanStack tool call metadata into the OpenInference tool-call shape.
 */
function toOpenInferenceToolCallFromTanStack(toolCall: TanStackToolCall): OpenInferenceToolCall {
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
function toOpenInferenceTools(tools: TanStackTool[]): OpenInferenceTool[] {
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
function getInvocationParameters(ctx: {
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
function getLLMInputValue(options: {
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
function getLLMOutput(options: {
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
 * Finalizes the current LLM span with output, usage, finish reason, and status.
 */
function endCurrentLLMSpan(options: {
  state: RequestState;
  status: { code: SpanStatusCode; message?: string };
  usage?: UsageInfo;
  finishReason?: string | null;
  error?: Error;
}) {
  const llmState = options.state.currentLLMSpan;
  if (llmState == null) {
    return;
  }

  const output = getLLMOutput({
    outputText: llmState.outputText,
    outputToolCalls: Array.from(llmState.outputToolCalls.values()),
  });

  if (output != null) {
    llmState.span.setAttributes({
      ...getOutputAttributes({ value: output.value, mimeType: output.mimeType }),
      ...getLLMAttributes({
        outputMessages: output.outputMessages,
      }),
    });
  }

  if (options.usage != null) {
    setUsageAttributes(llmState.span, options.usage);
  }

  if (options.finishReason != null) {
    llmState.span.setAttribute("tanstack.ai.finish_reason", options.finishReason);
  }

  if (options.error != null) {
    llmState.span.recordException(options.error);
  }

  llmState.span.setStatus(options.status);
  llmState.span.end();
  options.state.currentLLMSpan = undefined;
}

/**
 * Starts tracking a tool call emitted by the model so it can be reflected in
 * the LLM span output.
 */
function initializeToolCall(
  state: NonNullable<RequestState["currentLLMSpan"]>,
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
function updateToolCallArguments(
  state: NonNullable<RequestState["currentLLMSpan"]>,
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
function completeToolCall(
  state: NonNullable<RequestState["currentLLMSpan"]>,
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

/**
 * Creates a TanStack AI middleware that emits OpenInference-compatible AGENT,
 * LLM, and TOOL spans.
 *
 * The middleware is intentionally stateful per chat request so it can shape a
 * single agent run into a readable span tree while preserving LLM inputs and
 * outputs in the form Phoenix expects.
 */
export function openInferenceMiddleware({
  tracer,
  traceConfig,
}: OpenInferenceTanStackAIMiddlewareOptions = {}): ChatMiddleware {
  const oiTracer = new OITracer({
    tracer: tracer ?? trace.getTracer(INSTRUMENTATION_NAME, VERSION),
    traceConfig,
  });
  const requestStates = new Map<string, RequestState>();

  return {
    name: INSTRUMENTATION_NAME,

    onStart(ctx) {
      if (isTracingSuppressed(context.active())) {
        return;
      }

      const input = safelyJSONStringify({
        messages: ctx.messages,
        systemPrompts: ctx.systemPrompts,
        options: ctx.options,
        modelOptions: ctx.modelOptions,
        toolNames: ctx.toolNames,
        source: ctx.source,
        streaming: ctx.streaming,
      });

      const chatSpan = oiTracer.startSpan(AGENT_SPAN_NAME, {
        kind: SpanKind.INTERNAL,
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
          "tanstack.ai.request.id": ctx.requestId,
          "tanstack.ai.stream.id": ctx.streamId,
          "tanstack.ai.source": ctx.source,
          "tanstack.ai.streaming": ctx.streaming,
          ...(input == null ? {} : getInputAttributes({ value: input, mimeType: MimeType.JSON })),
          ...(ctx.options == null ? {} : getMetadataAttributes(ctx.options)),
        },
      });

      requestStates.set(ctx.requestId, {
        chatSpan,
        toolSpans: new Map<string, OISpan>(),
      });
    },

    onConfig(ctx, config) {
      if (ctx.phase !== "beforeModel" || isTracingSuppressed(context.active())) {
        return;
      }

      const state = requestStates.get(ctx.requestId);
      if (state == null) {
        return;
      }

      if (state.currentLLMSpan != null) {
        endCurrentLLMSpan({
          state,
          status: {
            code: SpanStatusCode.ERROR,
            message: "Starting next model call before previous LLM span finished",
          },
        });
      }

      const inputMessages = getInputMessages(config);
      const tools = toOpenInferenceTools(config.tools);
      const invocationParameters = getInvocationParameters({ model: ctx.model, config });
      const inputValue = getLLMInputValue({
        inputMessages,
        tools,
        invocationParameters,
      });

      const llmSpan = oiTracer.startSpan(
        `${LLM_SPAN_PREFIX}${ctx.iteration + 1}`,
        {
          kind: SpanKind.INTERNAL,
          attributes: {
            [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.LLM,
            ...(inputValue == null
              ? {}
              : getInputAttributes({ value: inputValue, mimeType: MimeType.JSON })),
            ...getLLMAttributes({
              provider: ctx.provider,
              system: ctx.provider,
              modelName: ctx.model,
              invocationParameters,
              inputMessages,
              tools,
            }),
          },
        },
        trace.setSpan(context.active(), state.chatSpan),
      );

      state.currentLLMSpan = {
        span: llmSpan,
        outputText: "",
        outputToolCalls: new Map<string, OpenInferenceToolCall>(),
      };
    },

    onIteration(ctx, info) {
      const chatSpan = requestStates.get(ctx.requestId)?.chatSpan;
      if (chatSpan == null) {
        return;
      }

      chatSpan.addEvent("tanstack.ai.iteration", {
        iteration: info.iteration,
        "tanstack.ai.message.id": info.messageId,
      });
    },

    onChunk(ctx, chunk) {
      const state = requestStates.get(ctx.requestId);
      if (state == null) {
        return;
      }

      const llmState = state.currentLLMSpan;
      if (llmState == null) {
        return;
      }

      switch (chunk.type) {
        case "TEXT_MESSAGE_CONTENT": {
          llmState.outputText = chunk.content ?? `${llmState.outputText}${chunk.delta}`;
          break;
        }
        case "TOOL_CALL_START": {
          initializeToolCall(llmState, chunk);
          break;
        }
        case "TOOL_CALL_ARGS": {
          updateToolCallArguments(llmState, chunk);
          break;
        }
        case "TOOL_CALL_END": {
          completeToolCall(llmState, chunk);
          break;
        }
        case "RUN_FINISHED": {
          endCurrentLLMSpan({
            state,
            status: { code: SpanStatusCode.OK },
            usage: chunk.usage,
            finishReason: chunk.finishReason,
          });
          break;
        }
        case "RUN_ERROR": {
          endCurrentLLMSpan({
            state,
            status: {
              code: SpanStatusCode.ERROR,
              message: chunk.error.message,
            },
            error: new Error(chunk.error.message),
          });
          break;
        }
      }
    },

    onBeforeToolCall(ctx, hookCtx) {
      if (isTracingSuppressed(context.active())) {
        return;
      }

      const state = requestStates.get(ctx.requestId);
      if (state == null) {
        return;
      }

      const serializedInput = safelyJSONStringify(hookCtx.args);
      const toolSpan = oiTracer.startSpan(
        `${TOOL_SPAN_PREFIX}${hookCtx.toolName}`,
        {
          kind: SpanKind.INTERNAL,
          attributes: {
            [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
            "tanstack.ai.tool.call.id": hookCtx.toolCallId,
            ...getToolAttributes({
              name: hookCtx.toolName,
              description: getToolDescription(hookCtx.tool),
              parameters: toRecord(hookCtx.args),
            }),
            ...(serializedInput == null
              ? {}
              : getInputAttributes({ value: serializedInput, mimeType: MimeType.JSON })),
          },
        },
        trace.setSpan(context.active(), state.chatSpan),
      );

      state.toolSpans.set(hookCtx.toolCallId, toolSpan);
    },

    onAfterToolCall(ctx, info) {
      const state = requestStates.get(ctx.requestId);
      const toolSpan = state?.toolSpans.get(info.toolCallId);
      if (toolSpan == null) {
        return;
      }

      toolSpan.setAttribute("tanstack.ai.tool.duration_ms", info.duration);

      if (info.ok) {
        const serializedOutput = safelyJSONStringify(info.result);
        toolSpan.setAttributes(
          serializedOutput == null
            ? {}
            : getOutputAttributes({ value: serializedOutput, mimeType: MimeType.JSON }),
        );
        toolSpan.setStatus({ code: SpanStatusCode.OK });
      } else {
        if (info.error instanceof Error) {
          toolSpan.recordException(info.error);
        }
        toolSpan.setStatus({
          code: SpanStatusCode.ERROR,
          message: info.error instanceof Error ? info.error.message : "Tool call failed",
        });
      }

      toolSpan.end();
      state?.toolSpans.delete(info.toolCallId);
    },

    onUsage() {},

    onFinish(ctx, info) {
      const state = requestStates.get(ctx.requestId);
      if (state == null) {
        return;
      }

      if (state.currentLLMSpan != null) {
        endCurrentLLMSpan({
          state,
          status: { code: SpanStatusCode.OK },
        });
      }

      state.chatSpan.setAttributes({
        ...getOutputAttributes(info.content),
        "tanstack.ai.finish_reason": info.finishReason ?? "",
        "tanstack.ai.duration_ms": info.duration,
      });

      state.chatSpan.setStatus({ code: SpanStatusCode.OK });
      state.toolSpans.forEach(finalizeSpan);
      finalizeSpan(state.chatSpan);
      requestStates.delete(ctx.requestId);
    },

    onAbort(ctx, info) {
      const state = requestStates.get(ctx.requestId);
      if (state == null) {
        return;
      }

      if (state.currentLLMSpan != null) {
        endCurrentLLMSpan({
          state,
          status: {
            code: SpanStatusCode.ERROR,
            message: info.reason ?? "TanStack AI run aborted",
          },
        });
      }

      state.chatSpan.setAttributes({
        "tanstack.ai.abort.reason": info.reason ?? "",
        "tanstack.ai.duration_ms": info.duration,
      });
      state.chatSpan.setStatus({
        code: SpanStatusCode.ERROR,
        message: info.reason ?? "TanStack AI run aborted",
      });
      endToolSpansWithError(state.toolSpans, info.reason ?? "TanStack AI run aborted");
      finalizeSpan(state.chatSpan);
      requestStates.delete(ctx.requestId);
    },

    onError(ctx, info) {
      const state = requestStates.get(ctx.requestId);
      if (state == null) {
        return;
      }

      if (state.currentLLMSpan != null) {
        endCurrentLLMSpan({
          state,
          status: {
            code: SpanStatusCode.ERROR,
            message: info.error instanceof Error ? info.error.message : "TanStack AI run failed",
          },
          error: info.error instanceof Error ? info.error : undefined,
        });
      }

      if (info.error instanceof Error) {
        state.chatSpan.recordException(info.error);
      }

      state.chatSpan.setAttributes({
        "tanstack.ai.duration_ms": info.duration,
      });
      state.chatSpan.setStatus({
        code: SpanStatusCode.ERROR,
        message: info.error instanceof Error ? info.error.message : "TanStack AI run failed",
      });
      endToolSpansWithError(
        state.toolSpans,
        info.error instanceof Error ? info.error.message : "TanStack AI run failed",
      );
      finalizeSpan(state.chatSpan);
      requestStates.delete(ctx.requestId);
    },
  };
}
