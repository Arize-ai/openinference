import type {
  SDKMessage,
  SDKResultMessage,
  SDKSession,
  SDKSessionOptions,
  SDKUserMessage,
} from "@anthropic-ai/claude-agent-sdk";
import type { Span } from "@opentelemetry/api";
import { context, SpanStatusCode, trace, TraceFlags } from "@opentelemetry/api";
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
            options,
            toolTracker,
            parentSpan: span,
          });
          const result = await original(message, modifiedOptions);

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
    const { session, delegatingTracker } = createInstrumentedSession(
      oiTracer,
      options,
      (modifiedOptions) => original(modifiedOptions),
    );
    return createSessionProxy(session, oiTracer, delegatingTracker);
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
    const { session, delegatingTracker } = createInstrumentedSession(
      oiTracer,
      options,
      (modifiedOptions) => original(sessionId, modifiedOptions),
    );
    return createSessionProxy(session, oiTracer, delegatingTracker);
  };
}

/**
 * A delegating ToolSpanTracker that forwards calls to whichever
 * per-turn tracker is currently active and injects the correct
 * parent span context for tool spans.
 */
interface DelegatingTracker extends ToolSpanTracker {
  /** Update the delegate target and parent span for the current turn. */
  setDelegate(tracker: ToolSpanTracker, parentSpan: Span): void;
  /** Clear the delegate (falls back to no-op). */
  clearDelegate(): void;
}

/**
 * Creates a delegating ToolSpanTracker, merges hooks into session options,
 * and calls the factory to produce the real SDK session.
 *
 * The delegating tracker forwards tool span operations to whichever per-turn
 * tracker is set via `setDelegate()`, allowing hooks registered at session
 * creation to route to the correct per-turn parent span.
 */
function createInstrumentedSession(
  oiTracer: OITracer,
  options: SDKSessionOptions,
  factory: (modifiedOptions: SDKSessionOptions) => SDKSession,
): { session: SDKSession; delegatingTracker: DelegatingTracker } {
  // Fallback tracker for calls that arrive before the first send()
  const fallbackTracker = new ToolSpanTracker(oiTracer);
  let activeTracker: ToolSpanTracker = fallbackTracker;
  let currentParentSpan: Span | undefined;

  // Build a delegating tracker that forwards to the per-turn active tracker.
  // `startToolSpan` injects the current turn span as the parent context so
  // tool spans are properly parented even when the SDK invokes hooks outside
  // our OTel context.with() scope.
  const delegatingTracker: DelegatingTracker = Object.create(fallbackTracker) as DelegatingTracker;
  delegatingTracker.setDelegate = (tracker: ToolSpanTracker, parentSpan: Span) => {
    activeTracker = tracker;
    currentParentSpan = parentSpan;
  };
  delegatingTracker.clearDelegate = () => {
    activeTracker = fallbackTracker;
    currentParentSpan = undefined;
  };
  delegatingTracker.startToolSpan = (
    toolName: string,
    toolInput: unknown,
    toolUseId: string,
    _parentContext?: ReturnType<typeof context.active>,
  ) => {
    // Override the parent context with the current turn span
    const parentCtx = currentParentSpan
      ? trace.setSpan(context.active(), currentParentSpan)
      : _parentContext;
    activeTracker.startToolSpan(toolName, toolInput, toolUseId, parentCtx);
  };
  delegatingTracker.endToolSpan = (...args: Parameters<ToolSpanTracker["endToolSpan"]>) =>
    activeTracker.endToolSpan(...args);
  delegatingTracker.endToolSpanWithError = (
    ...args: Parameters<ToolSpanTracker["endToolSpanWithError"]>
  ) => activeTracker.endToolSpanWithError(...args);
  delegatingTracker.endAllInFlight = () => activeTracker.endAllInFlight();

  // Create a non-recording sentinel span for hook registration.
  // The hooks themselves use the delegating tracker, so the sentinel is
  // only needed to satisfy the mergeHooks signature. Using a non-recording
  // span avoids polluting the exported trace with an internal-only span.
  const sentinelSpan = trace.wrapSpanContext({
    traceId: "0".repeat(32),
    spanId: "0".repeat(16),
    traceFlags: TraceFlags.NONE,
  });

  const modifiedOptions = mergeHooks({
    options,
    toolTracker: delegatingTracker,
    parentSpan: sentinelSpan,
  });

  const session = factory(modifiedOptions);
  return { session, delegatingTracker };
}

/**
 * Creates a Proxy around an SDKSession that:
 * - On `send(msg)`: Starts a per-turn AGENT span
 * - On `stream()`: Wraps the generator to process messages and end the span
 * - On `close()`: Ends any in-flight span
 *
 * **Hook injection:** Tool hooks are injected at session creation time via
 * `createInstrumentedSession`. Because the tracker and parent span change per
 * turn, the hooks use a delegating tracker whose target is updated on each
 * `send()`.
 *
 * **Sequential assumption:** The SDK's session API is inherently sequential —
 * `send()` queues a message and `stream()` drains it. Concurrent `send()`/`stream()`
 * calls on the same session are not supported by the SDK. The `currentTurnSpan` and
 * `currentToolTracker` shared state is therefore safe without synchronization.
 */
function createSessionProxy(
  session: SDKSession,
  oiTracer: OITracer,
  delegatingTracker: DelegatingTracker,
): SDKSession {
  let currentTurnSpan: Span | undefined;
  let currentToolTracker: ToolSpanTracker | undefined;
  let hasError = false;

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
            if (!hasError) {
              currentTurnSpan.setStatus({ code: SpanStatusCode.OK });
            }
            currentTurnSpan.end();
          }

          hasError = false;
          const inputAttrs = formatPromptAttributes(message);
          currentToolTracker = new ToolSpanTracker(oiTracer);

          currentTurnSpan = oiTracer.startSpan(`ClaudeAgent.turn`, {
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.AGENT,
              ...inputAttrs,
            },
          });

          delegatingTracker.setDelegate(currentToolTracker, currentTurnSpan);

          try {
            return await target.send(message);
          } catch (error) {
            currentToolTracker?.endAllInFlight();
            if (error instanceof Error) {
              currentTurnSpan.recordException(error);
              currentTurnSpan.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message,
              });
            }
            currentTurnSpan.end();
            currentTurnSpan = undefined;
            currentToolTracker = undefined;
            delegatingTracker.clearDelegate();
            throw error;
          }
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
            let spanEnded = false;
            try {
              for await (const msg of innerGen) {
                processSessionMessage(msg, span!);
                if (isResultErrorMessage(msg)) {
                  hasError = true;
                }
                yield msg;
              }
              // Stream completed normally
              toolTracker?.endAllInFlight();
              if (!hasError) {
                span!.setStatus({ code: SpanStatusCode.OK });
              }
              span!.end();
              spanEnded = true;
              currentTurnSpan = undefined;
              currentToolTracker = undefined;
              delegatingTracker.clearDelegate();
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
              spanEnded = true;
              currentTurnSpan = undefined;
              currentToolTracker = undefined;
              delegatingTracker.clearDelegate();
              throw error;
            } finally {
              // Handle early termination (break / return())
              if (!spanEnded && span) {
                toolTracker?.endAllInFlight();
                if (!hasError) {
                  span.setStatus({ code: SpanStatusCode.OK });
                }
                span.end();
                currentTurnSpan = undefined;
                currentToolTracker = undefined;
                delegatingTracker.clearDelegate();
              }
            }
          }

          return context.with(parentContext, () => wrappedGenerator());
        };
      }

      if (prop === "close") {
        return function wrappedClose(): void {
          if (currentTurnSpan) {
            currentToolTracker?.endAllInFlight();
            if (!hasError) {
              currentTurnSpan.setStatus({ code: SpanStatusCode.OK });
            }
            currentTurnSpan.end();
            currentTurnSpan = undefined;
            currentToolTracker = undefined;
            delegatingTracker.clearDelegate();
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
