import type {
  SDKMessage,
  SDKResultMessage,
  SDKSession,
  SDKSessionOptions,
  SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { Span } from "@opentelemetry/api";
import { context, SpanStatusCode, trace } from "@opentelemetry/api";
import { isTracingSuppressed } from "@opentelemetry/core";

import type { OITracer } from "@arizeai/openinference-core";
import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { ToolSpanTracker, mergeHooks } from "./hookInjector";
import {
  extractInitAttributes,
  extractResultErrorAttributes,
  extractResultSuccessAttributes,
  formatPromptAttributes,
  isResultErrorMessage,
  isResultSuccessMessage,
  isSystemInitMessage,
} from "./messageProcessor";

type CreateSessionFn = (options: SDKSessionOptions) => SDKSession;
type ResumeSessionFn = (sessionId: string, options: SDKSessionOptions) => SDKSession;
type PromptFn = (message: string, options: SDKSessionOptions) => Promise<SDKResultMessage>;

/**
 * Wraps the `unstable_v2_prompt()` function.
 * Creates a single AGENT span for the entire prompt -> result lifecycle.
 *
 * @param options.original - The original SDK `unstable_v2_prompt()` function
 * @param options.oiTracer - OITracer instance for creating spans
 * @returns A wrapped prompt function with identical signature
 */
export function wrapPrompt({
  original,
  oiTracer,
}: {
  original: PromptFn;
  oiTracer: OITracer;
}): PromptFn {
  return async function wrappedPrompt(
    message: string,
    options: SDKSessionOptions,
  ): Promise<SDKResultMessage> {
    const activeContext = context.active();
    if (isTracingSuppressed(activeContext)) {
      return original(message, options);
    }

    const inputAttrs = formatPromptAttributes(message);
    const toolTracker = new ToolSpanTracker(oiTracer);

    return oiTracer.startActiveSpan(
      `ClaudeAgent.prompt`,
      {
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
          ...inputAttrs,
        },
      },
      async (span: Span) => {
        try {
          const modifiedOptions = mergeHooks({
            options: options as Record<string, unknown>,
            toolTracker,
            parentSpan: span,
          });
          const result = await original(message, modifiedOptions as SDKSessionOptions);

          if (isResultSuccessMessage(result)) {
            span.setAttributes(extractResultSuccessAttributes(result));
            span.setStatus({ code: SpanStatusCode.OK });
          } else if (isResultErrorMessage(result)) {
            span.setAttributes(extractResultErrorAttributes(result));
            span.setStatus({
              code: SpanStatusCode.ERROR,
              message: `Result error: ${result.subtype}`,
            });
          }

          toolTracker.endAllInFlight();
          span.end();
          return result;
        } catch (error) {
          toolTracker.endAllInFlight();
          if (error instanceof Error) {
            span.recordException(error);
            span.setStatus({
              code: SpanStatusCode.ERROR,
              message: error.message,
            });
          }
          span.end();
          throw error;
        }
      },
    );
  };
}

/**
 * Wraps `unstable_v2_createSession()` to return a proxied session
 * that instruments `send()` + `stream()` with AGENT and TOOL spans.
 *
 * @param options.original - The original SDK `unstable_v2_createSession()` function
 * @param options.oiTracer - OITracer instance for creating spans
 * @returns A wrapped createSession function that returns an instrumented session
 */
export function wrapCreateSession({
  original,
  oiTracer,
}: {
  original: CreateSessionFn;
  oiTracer: OITracer;
}): CreateSessionFn {
  return function wrappedCreateSession(options: SDKSessionOptions): SDKSession {
    const session = original({ ...options });
    return createSessionProxy(session, oiTracer);
  };
}

/**
 * Wraps `unstable_v2_resumeSession()` to return a proxied session.
 *
 * @param options.original - The original SDK `unstable_v2_resumeSession()` function
 * @param options.oiTracer - OITracer instance for creating spans
 * @returns A wrapped resumeSession function that returns an instrumented session
 */
export function wrapResumeSession({
  original,
  oiTracer,
}: {
  original: ResumeSessionFn;
  oiTracer: OITracer;
}): ResumeSessionFn {
  return function wrappedResumeSession(sessionId: string, options: SDKSessionOptions): SDKSession {
    const session = original(sessionId, { ...options });
    return createSessionProxy(session, oiTracer);
  };
}

/**
 * Creates a Proxy around an SDKSession that:
 * - On `send(msg)`: Starts a per-turn AGENT span
 * - On `stream()`: Wraps the generator to process messages and end the span
 * - On `close()`: Ends any in-flight span
 */
function createSessionProxy(session: SDKSession, oiTracer: OITracer): SDKSession {
  let currentTurnSpan: Span | undefined;
  let currentToolTracker: ToolSpanTracker | undefined;

  return new Proxy(session, {
    get(target, prop, receiver) {
      if (prop === "send") {
        return async function wrappedSend(message: string | SDKUserMessage): Promise<void> {
          const activeContext = context.active();
          if (isTracingSuppressed(activeContext)) {
            return target.send(message);
          }

          // End any previous turn span
          if (currentTurnSpan) {
            currentToolTracker?.endAllInFlight();
            currentTurnSpan.setStatus({ code: SpanStatusCode.OK });
            currentTurnSpan.end();
          }

          const inputAttrs = formatPromptAttributes(message);
          currentToolTracker = new ToolSpanTracker(oiTracer);

          currentTurnSpan = oiTracer.startSpan(`ClaudeAgent.turn`, {
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
              ...inputAttrs,
            },
          });

          return target.send(message);
        };
      }

      if (prop === "stream") {
        return function wrappedStream(): AsyncGenerator<SDKMessage, void> {
          const innerGen = target.stream();
          const span = currentTurnSpan;
          const toolTracker = currentToolTracker;

          if (!span) {
            return innerGen;
          }

          const parentContext = trace.setSpan(context.active(), span);

          async function* wrappedGenerator(): AsyncGenerator<SDKMessage, void> {
            try {
              for await (const msg of innerGen) {
                processSessionMessage(msg, span!);
                yield msg;
              }
              // Stream completed
              toolTracker?.endAllInFlight();
              span!.setStatus({ code: SpanStatusCode.OK });
              span!.end();
              currentTurnSpan = undefined;
              currentToolTracker = undefined;
            } catch (error) {
              toolTracker?.endAllInFlight();
              if (error instanceof Error) {
                span!.recordException(error);
                span!.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
              }
              span!.end();
              currentTurnSpan = undefined;
              currentToolTracker = undefined;
              throw error;
            }
          }

          return context.with(parentContext, () => wrappedGenerator());
        };
      }

      if (prop === "close") {
        return function wrappedClose(): void {
          if (currentTurnSpan) {
            currentToolTracker?.endAllInFlight();
            currentTurnSpan.setStatus({ code: SpanStatusCode.OK });
            currentTurnSpan.end();
            currentTurnSpan = undefined;
            currentToolTracker = undefined;
          }
          return target.close();
        };
      }

      return Reflect.get(target, prop, receiver);
    },
  });
}

/**
 * Processes a message from the V2 session stream, setting span attributes.
 */
function processSessionMessage(msg: SDKMessage, span: Span): void {
  if (isSystemInitMessage(msg)) {
    const { sessionId, model } = extractInitAttributes(msg);
    span.setAttributes({
      [SemanticConventions.SESSION_ID]: sessionId,
      [SemanticConventions.LLM_MODEL_NAME]: model,
    });
  } else if (isResultSuccessMessage(msg)) {
    span.setAttributes(extractResultSuccessAttributes(msg));
  } else if (isResultErrorMessage(msg)) {
    span.setAttributes(extractResultErrorAttributes(msg));
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: `Result error: ${msg.subtype}`,
    });
  }
}
