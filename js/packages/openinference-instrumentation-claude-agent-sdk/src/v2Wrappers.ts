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
  extractResultErrorAttributes,
  extractResultSuccessAttributes,
  formatPromptAttributes,
  isResultErrorMessage,
  isResultSuccessMessage,
  isSystemInitMessage,
} from "./messageProcessor";

/**
 * Structural types matching the SDK's session-based V2 API.
 * These use runtime shape checks, not imported types.
 */
interface SDKSessionLike {
  readonly sessionId: string;
  send(message: string | Record<string, unknown>): Promise<void>;
  stream(): AsyncGenerator<unknown, void>;
  close(): void;
}

type SDKSessionOptions = Record<string, unknown>;

type CreateSessionFn = (options: SDKSessionOptions) => SDKSessionLike;
type ResumeSessionFn = (sessionId: string, options: SDKSessionOptions) => SDKSessionLike;
type PromptFn = (message: string, options: SDKSessionOptions) => Promise<unknown>;

/**
 * Wraps the `unstable_v2_prompt()` function.
 * Creates a single AGENT span for the entire prompt -> result lifecycle.
 */
export function wrapPrompt(original: PromptFn, oiTracer: OITracer): PromptFn {
  return async function wrappedPrompt(
    message: string,
    options: SDKSessionOptions,
  ): Promise<unknown> {
    const activeContext = context.active();
    if (isTracingSuppressed(activeContext)) {
      return original(message, options);
    }

    const { inputValue, inputMimeType } = formatPromptAttributes(message);
    const toolTracker = new ToolSpanTracker(oiTracer);

    return oiTracer.startActiveSpan(
      `Claude Agent SDK prompt`,
      {
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
          [SemanticConventions.INPUT_VALUE]: inputValue,
          [SemanticConventions.INPUT_MIME_TYPE]: inputMimeType,
        },
      },
      async (span: Span) => {
        try {
          const modifiedOptions = mergeHooks(options, toolTracker, span);
          const result = await original(message, modifiedOptions);

          if (isResultSuccessMessage(result)) {
            span.setAttributes(extractResultSuccessAttributes(result));
            span.setStatus({ code: SpanStatusCode.OK });
          } else if (isResultErrorMessage(result)) {
            span.setAttributes(extractResultErrorAttributes(result));
            span.setStatus({
              code: SpanStatusCode.ERROR,
              message: `Result error: ${(result as unknown as Record<string, unknown>).subtype}`,
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
 */
export function wrapCreateSession(original: CreateSessionFn, oiTracer: OITracer): CreateSessionFn {
  return function wrappedCreateSession(options: SDKSessionOptions): SDKSessionLike {
    const session = original({ ...options });
    return createSessionProxy(session, oiTracer);
  };
}

/**
 * Wraps `unstable_v2_resumeSession()` to return a proxied session.
 */
export function wrapResumeSession(original: ResumeSessionFn, oiTracer: OITracer): ResumeSessionFn {
  return function wrappedResumeSession(
    sessionId: string,
    options: SDKSessionOptions,
  ): SDKSessionLike {
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
function createSessionProxy(session: SDKSessionLike, oiTracer: OITracer): SDKSessionLike {
  let currentTurnSpan: Span | undefined;
  let currentToolTracker: ToolSpanTracker | undefined;

  return new Proxy(session, {
    get(target, prop, receiver) {
      if (prop === "send") {
        return async function wrappedSend(
          message: string | Record<string, unknown>,
        ): Promise<void> {
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

          const { inputValue, inputMimeType } = formatPromptAttributes(message);
          currentToolTracker = new ToolSpanTracker(oiTracer);

          currentTurnSpan = oiTracer.startSpan(`Claude Agent SDK turn`, {
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
              [SemanticConventions.INPUT_VALUE]: inputValue,
              [SemanticConventions.INPUT_MIME_TYPE]: inputMimeType,
              [SemanticConventions.SESSION_ID]: target.sessionId,
            },
          });

          return target.send(message);
        };
      }

      if (prop === "stream") {
        return function wrappedStream(): AsyncGenerator<unknown, void> {
          const innerGen = target.stream();
          const span = currentTurnSpan;
          const toolTracker = currentToolTracker;

          if (!span) {
            return innerGen;
          }

          const parentContext = trace.setSpan(context.active(), span);

          async function* wrappedGenerator(): AsyncGenerator<unknown, void> {
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
function processSessionMessage(msg: unknown, span: Span): void {
  if (isSystemInitMessage(msg)) {
    const record = msg as unknown as Record<string, unknown>;
    span.setAttribute(SemanticConventions.SESSION_ID, record.session_id as string);
    span.setAttribute(SemanticConventions.LLM_MODEL_NAME, record.model as string);
  } else if (isResultSuccessMessage(msg)) {
    span.setAttributes(extractResultSuccessAttributes(msg));
  } else if (isResultErrorMessage(msg)) {
    span.setAttributes(extractResultErrorAttributes(msg));
    span.setStatus({
      code: SpanStatusCode.ERROR,
      message: `Result error: ${(msg as unknown as Record<string, unknown>).subtype}`,
    });
  }
}
