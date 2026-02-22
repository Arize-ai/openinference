import type { Span } from "@opentelemetry/api";
import { context, SpanStatusCode, trace } from "@opentelemetry/api";

import type { OITracer } from "@arizeai/openinference-core";
import { safelyJSONStringify } from "@arizeai/openinference-core";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

/**
 * Structural type for the hooks option in SDK Options/SDKSessionOptions.
 * Matches: Partial<Record<HookEvent, HookCallbackMatcher[]>>
 */
type HooksOption = Record<string, HookCallbackMatcher[]>;

/**
 * Structural type matching SDK's HookCallbackMatcher interface.
 */
interface HookCallbackMatcher {
  matcher?: string;
  hooks: HookCallback[];
  timeout?: number;
}

/**
 * Structural type matching SDK's HookCallback function.
 */
type HookCallback = (
  input: Record<string, unknown>,
  toolUseID: string | undefined,
  options: { signal: AbortSignal },
) => Promise<Record<string, unknown>>;

/**
 * Tracks in-flight tool spans, correlating PreToolUse → PostToolUse/PostToolUseFailure
 * via tool_use_id.
 */
export class ToolSpanTracker {
  private inFlightSpans = new Map<string, Span>();
  private oiTracer: OITracer;

  constructor(oiTracer: OITracer) {
    this.oiTracer = oiTracer;
  }

  /**
   * Starts a TOOL span for the given tool invocation.
   * @param toolName Name of the tool being invoked
   * @param toolInput The tool's input parameters
   * @param toolUseId Unique identifier correlating Pre/Post hooks
   * @param parentContext The parent context to create the span under
   */
  startToolSpan(
    toolName: string,
    toolInput: unknown,
    toolUseId: string,
    parentContext?: ReturnType<typeof context.active>,
  ): void {
    const inputStr = safelyJSONStringify(toolInput) ?? "";
    const ctx = parentContext ?? context.active();
    const span = this.oiTracer.startSpan(
      `Tool: ${toolName}`,
      {
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
          [SemanticConventions.TOOL_NAME]: toolName,
          [SemanticConventions.TOOL_PARAMETERS]: inputStr,
          [SemanticConventions.INPUT_VALUE]: inputStr,
          [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
        },
      },
      ctx,
    );
    this.inFlightSpans.set(toolUseId, span);
  }

  /**
   * Ends a TOOL span successfully with the tool's response.
   */
  endToolSpan(toolUseId: string, toolResponse?: unknown): void {
    const span = this.inFlightSpans.get(toolUseId);
    if (!span) return;
    this.inFlightSpans.delete(toolUseId);

    if (toolResponse !== undefined) {
      const outputStr = safelyJSONStringify(toolResponse) ?? "";
      span.setAttributes({
        [SemanticConventions.OUTPUT_VALUE]: outputStr,
        [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
      });
    }
    span.setStatus({ code: SpanStatusCode.OK });
    span.end();
  }

  /**
   * Ends a TOOL span with an error.
   */
  endToolSpanWithError(toolUseId: string, error: string): void {
    const span = this.inFlightSpans.get(toolUseId);
    if (!span) return;
    this.inFlightSpans.delete(toolUseId);

    span.setStatus({ code: SpanStatusCode.ERROR, message: error });
    span.recordException(new Error(error));
    span.end();
  }

  /**
   * Ends all in-flight tool spans. Safety cleanup for abandoned generators.
   */
  endAllInFlight(): void {
    for (const [toolUseId, span] of this.inFlightSpans) {
      span.setStatus({ code: SpanStatusCode.ERROR, message: "Abandoned" });
      span.end();
      this.inFlightSpans.delete(toolUseId);
    }
  }
}

/**
 * Creates hook callback matchers for PreToolUse, PostToolUse, and PostToolUseFailure
 * that track tool spans via the provided ToolSpanTracker.
 *
 * Returns an empty SyncHookJSONOutput ({}) so our hooks never affect tool execution.
 */
function createToolHookMatchers(
  toolTracker: ToolSpanTracker,
  parentSpan: Span,
): Record<string, HookCallbackMatcher[]> {
  const parentContext = trace.setSpan(context.active(), parentSpan);

  const preToolUse: HookCallbackMatcher = {
    hooks: [
      async (input: Record<string, unknown>) => {
        const toolName = typeof input.tool_name === "string" ? input.tool_name : "unknown";
        const toolInput = input.tool_input;
        const toolUseId = typeof input.tool_use_id === "string" ? input.tool_use_id : "";
        if (toolUseId) {
          toolTracker.startToolSpan(toolName, toolInput, toolUseId, parentContext);
        }
        return {};
      },
    ],
  };

  const postToolUse: HookCallbackMatcher = {
    hooks: [
      async (input: Record<string, unknown>) => {
        const toolUseId = typeof input.tool_use_id === "string" ? input.tool_use_id : "";
        if (toolUseId) {
          toolTracker.endToolSpan(toolUseId, input.tool_response);
        }
        return {};
      },
    ],
  };

  const postToolUseFailure: HookCallbackMatcher = {
    hooks: [
      async (input: Record<string, unknown>) => {
        const toolUseId = typeof input.tool_use_id === "string" ? input.tool_use_id : "";
        const error = typeof input.error === "string" ? input.error : "Unknown error";
        if (toolUseId) {
          toolTracker.endToolSpanWithError(toolUseId, error);
        }
        return {};
      },
    ],
  };

  return {
    PreToolUse: [preToolUse],
    PostToolUse: [postToolUse],
    PostToolUseFailure: [postToolUseFailure],
  };
}

/**
 * Merges our tool-tracking hooks into an existing options object,
 * preserving any user-defined hooks by appending our matchers.
 *
 * Returns a new options object (does not mutate the original).
 */
export function mergeHooks(
  options: Record<string, unknown> | undefined,
  toolTracker: ToolSpanTracker,
  parentSpan: Span,
): Record<string, unknown> {
  const opts = options ?? {};
  const existingHooks = (opts.hooks ?? {}) as HooksOption;
  const ourHooks = createToolHookMatchers(toolTracker, parentSpan);

  const mergedHooks: HooksOption = { ...existingHooks };
  for (const [event, matchers] of Object.entries(ourHooks)) {
    const existing = mergedHooks[event] ?? [];
    mergedHooks[event] = [...existing, ...matchers];
  }

  return { ...opts, hooks: mergedHooks };
}
