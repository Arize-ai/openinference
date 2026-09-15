import type { HookCallback, HookCallbackMatcher, HookEvent } from "@anthropic-ai/claude-agent-sdk";
import type { Span } from "@opentelemetry/api";
import { context, diag, SpanStatusCode, trace } from "@opentelemetry/api";

import type { OITracer } from "@arizeai/openinference-core";
import {
  getInputAttributes,
  getOutputAttributes,
  getToolAttributes,
  isObjectWithStringKeys,
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
  return isObjectWithStringKeys(value) ? value : {};
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
 * The hook events this instrumentation registers matchers for. Single source of truth
 * shared by {@link createToolHookMatchers} (which returns exactly these keys) and
 * {@link mergeHooks} (which merges exactly these keys).
 */
const TOOL_HOOK_EVENTS = [
  "PreToolUse",
  "PostToolUse",
  "PostToolUseFailure",
] as const satisfies readonly HookEvent[];

type ToolHookEvent = (typeof TOOL_HOOK_EVENTS)[number];

/**
 * Creates hook callback matchers for PreToolUse, PostToolUse, and PostToolUseFailure
 * that track tool spans via the provided ToolSpanTracker.
 *
 * Returns an empty SyncHookJSONOutput ({}) so our hooks never affect tool execution.
 */
function createToolHookMatchers(
  toolTracker: ToolSpanTracker,
  parentSpan: Span,
): Record<ToolHookEvent, HookCallbackMatcher[]> {
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
export function mergeHooks<T extends { hooks?: HooksOption }>(args: {
  options: T | undefined;
  toolTracker: ToolSpanTracker;
  parentSpan: Span;
}): T & { hooks: HooksOption };
export function mergeHooks({
  options,
  toolTracker,
  parentSpan,
}: {
  options: { hooks?: HooksOption } | undefined;
  toolTracker: ToolSpanTracker;
  parentSpan: Span;
}): { hooks: HooksOption } {
  const opts = options ?? {};
  const existingHooks = opts.hooks ?? {};
  const ourHooks = createToolHookMatchers(toolTracker, parentSpan);

  const mergedHooks: HooksOption = { ...existingHooks };
  for (const event of TOOL_HOOK_EVENTS) {
    const matchers = ourHooks[event];
    const existing = mergedHooks[event] ?? [];
    mergedHooks[event] = [...existing, ...matchers];
  }

  return { ...opts, hooks: mergedHooks };
}
