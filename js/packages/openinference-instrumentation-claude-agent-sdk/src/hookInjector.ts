import type { HookCallback, HookCallbackMatcher, HookEvent } from "@anthropic-ai/claude-agent-sdk";
import type { Span } from "@opentelemetry/api";
import { context, diag, SpanStatusCode, trace } from "@opentelemetry/api";

import type { OITracer } from "@arizeai/openinference-core";
import {
  getInputAttributes,
  getOutputAttributes,
  getToolAttributes,
  safelyJSONStringify,
} from "@arizeai/openinference-core";
import {
  MimeType,
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

/**
 * Hook configuration from SDK options, keyed by hook event name.
 */
type HooksOption = Partial<Record<HookEvent, HookCallbackMatcher[]>>;

/**
 * Safely coerces an unknown value to Record<string, unknown>.
 * Returns an empty object for non-object values (strings, arrays, null, etc.).
 */
function asRecord(value: unknown): Record<string, unknown> {
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

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
      `${toolName}`,
      {
        attributes: {
          [SemanticConventions.OPENINFERENCE_SPAN_KIND]: OpenInferenceSpanKind.TOOL,
          ...getToolAttributes({ name: toolName, parameters: asRecord(toolInput) }),
          ...getInputAttributes({ value: inputStr, mimeType: MimeType.JSON }),
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
      span.setAttributes(getOutputAttributes({ value: outputStr, mimeType: MimeType.JSON }));
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
): Partial<Record<HookEvent, HookCallbackMatcher[]>> {
  const parentContext = trace.setSpan(context.active(), parentSpan);

  const preToolUseHook: HookCallback = async (input) => {
    try {
      if (input.hook_event_name !== "PreToolUse") return {};
      const { tool_name, tool_input, tool_use_id } = input;
      toolTracker.startToolSpan(tool_name, tool_input, tool_use_id, parentContext);
    } catch (e) {
      diag.warn("OpenInference: PreToolUse hook error", e);
    }
    return {};
  };

  const postToolUseHook: HookCallback = async (input) => {
    try {
      if (input.hook_event_name !== "PostToolUse") return {};
      const { tool_use_id, tool_response } = input;
      toolTracker.endToolSpan(tool_use_id, tool_response);
    } catch (e) {
      diag.warn("OpenInference: PostToolUse hook error", e);
    }
    return {};
  };

  const postToolUseFailureHook: HookCallback = async (input) => {
    try {
      if (input.hook_event_name !== "PostToolUseFailure") return {};
      const { tool_use_id, error } = input;
      toolTracker.endToolSpanWithError(tool_use_id, error);
    } catch (e) {
      diag.warn("OpenInference: PostToolUseFailure hook error", e);
    }
    return {};
  };

  return {
    PreToolUse: [{ hooks: [preToolUseHook] }],
    PostToolUse: [{ hooks: [postToolUseHook] }],
    PostToolUseFailure: [{ hooks: [postToolUseFailureHook] }],
  };
}

/**
 * Merges our tool-tracking hooks into an existing options object,
 * preserving any user-defined hooks by appending our matchers.
 *
 * Returns a new options object (does not mutate the original).
 */
export function mergeHooks<T extends { hooks?: HooksOption }>({
  options,
  toolTracker,
  parentSpan,
}: {
  options: T | undefined;
  toolTracker: ToolSpanTracker;
  parentSpan: Span;
}): T {
  const opts = options ?? ({} as T);
  const existingHooks = opts.hooks ?? {};
  const ourHooks = createToolHookMatchers(toolTracker, parentSpan);

  const mergedHooks: HooksOption = { ...existingHooks };
  for (const [event, matchers] of Object.entries(ourHooks)) {
    const existing = mergedHooks[event as HookEvent] ?? [];
    mergedHooks[event as HookEvent] = [...existing, ...matchers];
  }

  return { ...opts, hooks: mergedHooks };
}
