import { context, SpanKind, SpanStatusCode, trace } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";
import type { ChatMiddleware } from "@tanstack/ai";

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

import {
  AGENT_SPAN_NAME,
  INSTRUMENTATION_NAME,
  LLM_SPAN_PREFIX,
  TOOL_SPAN_PREFIX,
} from "./constants";
import {
  completeToolCall,
  getInputMessages,
  getInvocationParameters,
  getLLMInputValue,
  getLLMOutput,
  initializeToolCall,
  setUsageAttributes,
  toOpenInferenceTools,
  updateToolCallArguments,
} from "./converters";
import type { OpenInferenceTanStackAIMiddlewareOptions, RequestState, UsageInfo } from "./types";
import { finalizeSpan, getToolDescription, toRecord } from "./utils";
import { VERSION } from "./version";

/**
 * Marks unfinished tool spans as failed before ending them.
 */
function endToolSpansWithError(
  toolSpans: Map<string, ReturnType<OITracer["startSpan"]>>,
  message: string,
) {
  toolSpans.forEach((span) => {
    span.setStatus({ code: SpanStatusCode.ERROR, message });
    span.end();
  });
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
      ...getLLMAttributes({ outputMessages: output.outputMessages }),
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
        toolSpans: new Map(),
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
      const inputValue = getLLMInputValue({ inputMessages, tools, invocationParameters });

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
        outputToolCalls: new Map(),
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
            usage: chunk.usage ?? undefined,
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

export type { OpenInferenceTanStackAIMiddlewareOptions } from "./types";
