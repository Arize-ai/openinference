import type { AttributeValue, HrTime } from "@opentelemetry/api";
import { SpanStatusCode } from "@opentelemetry/api";
import type { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";

import { safelyJSONStringify } from "@arizeai/openinference-core";
import { MimeType, SemanticConventions } from "@arizeai/openinference-semantic-conventions";

import {
  addOpenInferenceAttributesToSpan,
  ensureAgentSpanKind,
  isAISDKSpanByName,
  promoteSpanToRoot,
} from "./utils";

const LLM_INPUT_MESSAGES = "llm.input_messages";
const LLM_OUTPUT_MESSAGES = "llm.output_messages";

/** A captured set of input (or output) attributes plus the time used to order it. */
type TimedAttributes = {
  attributes: Record<string, AttributeValue>;
  /** Nanosecond timestamp used to pick the earliest input / latest output. */
  nanos: number;
};

type TraceAggregate = {
  activeSpans: number;
  hadError: boolean;
  firstErrorMessage?: string;
  isAISDKTrace: boolean;
  /**
   * Span id of the span promoted to root for this trace (the first AI SDK span
   * to start). Only set when `agentTraceMode` is enabled. Undefined until the
   * first AI SDK span starts.
   */
  promotedRootSpanId?: string;
  /**
   * The promoted root span, held back from export until the trace completes.
   * A framework wrapper root (e.g. `ai.eve.turn`) often ends *before* its child
   * spans, so we defer exporting it until the last span ends and then stamp the
   * aggregated input/output onto it.
   */
  pendingRoot?: ReadableSpan;
  /** Earliest child input attributes, stamped onto the promoted root if it has none. */
  firstInput?: TimedAttributes;
  /** Latest child output attributes, stamped onto the promoted root if it has none. */
  lastOutput?: TimedAttributes;
};

const hrTimeToNanos = (time: HrTime): number => time[0] * 1e9 + time[1];

// Input/output can be represented either as the generic OpenInference value
// attributes (`input.value` / `output.value`, used by the Vercel AI SDK spans)
// or as flattened message attributes (`llm.input_messages.*` /
// `llm.output_messages.*`, used by GenAI-semconv spans such as Eve's). Capture
// whichever form a child carries so a wrapper root can display it.
const INPUT_VALUE_PREFIXES = [SemanticConventions.INPUT_VALUE, LLM_INPUT_MESSAGES];
const OUTPUT_VALUE_PREFIXES = [SemanticConventions.OUTPUT_VALUE, LLM_OUTPUT_MESSAGES];
const INPUT_COPY_PREFIXES = [
  SemanticConventions.INPUT_VALUE,
  SemanticConventions.INPUT_MIME_TYPE,
  LLM_INPUT_MESSAGES,
];
const OUTPUT_COPY_PREFIXES = [
  SemanticConventions.OUTPUT_VALUE,
  SemanticConventions.OUTPUT_MIME_TYPE,
  LLM_OUTPUT_MESSAGES,
];

const matchesPrefix = (key: string, prefix: string): boolean =>
  key === prefix || key.startsWith(`${prefix}.`);

const hasAnyPrefix = (span: ReadableSpan, prefixes: string[]): boolean =>
  Object.keys(span.attributes).some((k) => prefixes.some((p) => matchesPrefix(k, p)));

const collectByPrefixes = (
  span: ReadableSpan,
  prefixes: string[],
): Record<string, AttributeValue> => {
  const out: Record<string, AttributeValue> = {};
  for (const [k, v] of Object.entries(span.attributes)) {
    if (v != null && prefixes.some((p) => matchesPrefix(k, p))) {
      out[k] = v;
    }
  }
  return out;
};

const isLikelyAISDKSpan = (span: ReadableSpan | Span): boolean => {
  const attrs = span.attributes as Record<string, unknown> | undefined;
  const opName = attrs?.["operation.name"];
  const opId = attrs?.["ai.operationId"];

  if (typeof opName === "string" && opName.startsWith("ai.")) return true;
  if (typeof opId === "string" && opId.startsWith("ai.")) return true;

  // gen_ai.* indicates AI SDK v6+ GenAI spans
  return attrs != null && Object.keys(attrs).some((k) => k.startsWith("gen_ai."));
};

const spanHasErrorSignal = (span: ReadableSpan): { error: boolean; message?: string } => {
  if (span.status.code === SpanStatusCode.ERROR) {
    return { error: true, message: span.status.message };
  }

  const attrs = span.attributes as Record<string, unknown>;
  const finishReason = attrs["ai.response.finishReason"];
  if (finishReason === "error") {
    return { error: true, message: "ai.response.finishReason=error" };
  }

  const genFinishReasons = attrs["gen_ai.response.finish_reasons"];
  if (Array.isArray(genFinishReasons) && genFinishReasons.includes("error")) {
    return {
      error: true,
      message: "gen_ai.response.finish_reasons includes error",
    };
  }

  const hasExceptionEvent = span.events?.some((e) => e.name === "exception");
  if (hasExceptionEvent) {
    return { error: true, message: "exception" };
  }

  return { error: false };
};

const maybeSetRootStatus = (span: ReadableSpan, agg: TraceAggregate): void => {
  // Only set status on the root span, and only when it's currently UNSET.
  if (span.parentSpanId != null) return;
  if (!isLikelyAISDKSpan(span)) return;
  if (span.status.code !== SpanStatusCode.UNSET) return;

  // ReadableSpan is typed as readonly; runtime Span objects are mutable.
  (span as unknown as { status: { code: SpanStatusCode; message?: string } }).status = agg.hadError
    ? { code: SpanStatusCode.ERROR, message: agg.firstErrorMessage }
    : { code: SpanStatusCode.OK };
};

const maybeSetSpanOkStatus = (span: ReadableSpan): void => {
  // Set OK status on AI SDK spans that completed without error
  if (!isLikelyAISDKSpan(span)) return;
  if (span.status.code !== SpanStatusCode.UNSET) return;

  // ReadableSpan is typed as readonly; runtime Span objects are mutable.
  (span as unknown as { status: { code: SpanStatusCode; message?: string } }).status = {
    code: SpanStatusCode.OK,
  };
};

const maybeRenameRootSpan = (span: ReadableSpan): void => {
  if (span.parentSpanId != null) return;
  if (!isLikelyAISDKSpan(span)) return;

  const attrs = span.attributes as Record<string, unknown>;
  const operationName = attrs["operation.name"];
  if (typeof operationName !== "string" || operationName.length === 0) return;
  if (span.name === operationName) return;

  // NOTE: Span.updateName() refuses to update after end(); by the time span processors
  // run, spans are already ended. Assign directly.
  (span as unknown as { name: string }).name = operationName;
};

/**
 * Records this span's input as the trace's earliest input and its output as the
 * trace's latest output, ordered by span start/end time. Used to derive the
 * input/output shown on a promoted root span (e.g. a framework wrapper) that
 * carries no input/output of its own.
 */
const captureChildIO = (span: ReadableSpan, agg: TraceAggregate): void => {
  if (hasAnyPrefix(span, INPUT_VALUE_PREFIXES)) {
    const nanos = hrTimeToNanos(span.startTime);
    if (agg.firstInput == null || nanos < agg.firstInput.nanos) {
      agg.firstInput = { attributes: collectByPrefixes(span, INPUT_COPY_PREFIXES), nanos };
    }
  }

  if (hasAnyPrefix(span, OUTPUT_VALUE_PREFIXES)) {
    const nanos = hrTimeToNanos(span.endTime);
    if (agg.lastOutput == null || nanos > agg.lastOutput.nanos) {
      agg.lastOutput = { attributes: collectByPrefixes(span, OUTPUT_COPY_PREFIXES), nanos };
    }
  }
};

/** Recursively turns objects with all-numeric keys into arrays. */
const numericKeyedToArrays = (node: unknown): unknown => {
  if (node == null || typeof node !== "object") return node;
  const obj = node as Record<string, unknown>;
  const keys = Object.keys(obj);
  if (keys.length > 0 && keys.every((k) => /^\d+$/.test(k))) {
    const arr: unknown[] = [];
    for (const k of keys.sort((a, b) => Number(a) - Number(b))) {
      arr[Number(k)] = numericKeyedToArrays(obj[k]);
    }
    return arr;
  }
  const out: Record<string, unknown> = {};
  for (const k of keys) out[k] = numericKeyedToArrays(obj[k]);
  return out;
};

/**
 * Reconstructs the nested value held by flattened OpenInference attributes under
 * `prefix` (e.g. `llm.input_messages.0.message.content`) back into a structured
 * array/object.
 */
const unflattenAttributes = (attrs: Record<string, AttributeValue>, prefix: string): unknown => {
  const tree: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(attrs)) {
    if (!key.startsWith(`${prefix}.`)) continue;
    const path = key.slice(prefix.length + 1).split(".");
    let cur = tree;
    for (let i = 0; i < path.length - 1; i++) {
      const seg = path[i];
      if (cur[seg] == null || typeof cur[seg] !== "object") cur[seg] = {};
      cur = cur[seg] as Record<string, unknown>;
    }
    cur[path[path.length - 1]] = value;
  }
  return numericKeyedToArrays(tree);
};

/**
 * Derives a scalar `*.value` (+ JSON mime) from flattened `llm.*_messages`
 * attributes when no scalar value is present, so the value renders in tools that
 * key off `input.value` / `output.value` (e.g. Phoenix/AX panels). This fills the
 * gap for GenAI-semconv wrapper roots whose children expose I/O only as messages;
 * a value copied from a child (e.g. the Vercel AI SDK's `input.value`) is kept.
 */
const deriveValueFromMessages = (
  attrs: Record<string, AttributeValue>,
  messagesPrefix: string,
  valueKey: string,
  mimeKey: string,
): void => {
  if (attrs[valueKey] != null) return;
  const hasMessages = Object.keys(attrs).some((k) => k.startsWith(`${messagesPrefix}.`));
  if (!hasMessages) return;
  const json = safelyJSONStringify(unflattenAttributes(attrs, messagesPrefix));
  if (json != null) {
    attrs[valueKey] = json;
    attrs[mimeKey] = MimeType.JSON;
  }
};

/**
 * Stamps the trace's aggregated input/output onto the promoted root span, but
 * only for sides it does not already have. A root that is itself a top-level
 * AI SDK call (e.g. `ai.streamText`) already carries its own input/output and is
 * left untouched. For framework wrappers whose children carry I/O as
 * `llm.*_messages` (e.g. Eve's `gen_ai` spans), a scalar `input.value` /
 * `output.value` is derived from those messages so the root renders correctly.
 */
const applyAggregatedIO = (span: ReadableSpan, agg: TraceAggregate): void => {
  const attrs = span.attributes as Record<string, AttributeValue>;
  if (agg.firstInput != null && !hasAnyPrefix(span, INPUT_VALUE_PREFIXES)) {
    Object.assign(attrs, agg.firstInput.attributes);
    deriveValueFromMessages(
      attrs,
      LLM_INPUT_MESSAGES,
      SemanticConventions.INPUT_VALUE,
      SemanticConventions.INPUT_MIME_TYPE,
    );
  }
  if (agg.lastOutput != null && !hasAnyPrefix(span, OUTPUT_VALUE_PREFIXES)) {
    Object.assign(attrs, agg.lastOutput.attributes);
    deriveValueFromMessages(
      attrs,
      LLM_OUTPUT_MESSAGES,
      SemanticConventions.OUTPUT_VALUE,
      SemanticConventions.OUTPUT_MIME_TYPE,
    );
  }
};

/**
 * Manages trace-level aggregate state for error propagation and span tracking.
 *
 * Tracking strategy to prevent memory leaks:
 * - All spans are tracked at onStart (we need the count for cleanup)
 * - At onEnd, we check if the span is an AI SDK span and mark the trace accordingly
 * - Error state is only aggregated for AI SDK traces
 * - Non-AI SDK traces are cleaned up immediately when they complete
 */
export class TraceAggregateManager {
  private readonly traceAggregates = new Map<string, TraceAggregate>();
  private readonly agentTraceMode: boolean;

  /**
   * @param options.agentTraceMode - when true, the manager selects the first AI
   * SDK span per trace as the promoted root, defers its export until the trace
   * completes, and stamps the aggregated child input/output onto it. See the
   * processor's `agentTraceMode` option.
   */
  constructor({ agentTraceMode = false }: { agentTraceMode?: boolean } = {}) {
    this.agentTraceMode = agentTraceMode;
  }

  /**
   * Track a span starting. All spans are tracked to maintain accurate counts.
   * In agentTraceMode, the first AI SDK span to start in a trace is recorded as
   * the span to promote to root (parents always start before their children, so
   * the first AI SDK span is the outermost one).
   */
  onStart(span: Span): void {
    const traceId = span.spanContext().traceId;
    const agg = this.traceAggregates.get(traceId);
    if (agg) {
      agg.activeSpans += 1;
    } else {
      this.traceAggregates.set(traceId, {
        activeSpans: 1,
        hadError: false,
        isAISDKTrace: false,
      });
    }

    if (this.agentTraceMode) {
      const current = this.traceAggregates.get(traceId);
      if (current != null && current.promotedRootSpanId == null && isAISDKSpanByName(span)) {
        current.promotedRootSpanId = span.spanContext().spanId;
      }
    }
  }

  /**
   * Process a span ending. Returns the spans that should be exported as a result
   * (normally just the span itself). In agentTraceMode the promoted root is held
   * back until the trace completes, so:
   * - the root's own onEnd may return `[]` (deferred), and
   * - the last span's onEnd returns `[span, root]` (the deferred root, finalized).
   */
  onEnd(span: ReadableSpan): ReadableSpan[] {
    addOpenInferenceAttributesToSpan(span);

    const traceId = span.spanContext().traceId;
    const agg = this.traceAggregates.get(traceId);

    // If we don't have an aggregate for this trace, just process the span
    if (agg == null) {
      maybeRenameRootSpan(span);
      return [span];
    }

    const isPromotedRoot =
      this.agentTraceMode && agg.promotedRootSpanId === span.spanContext().spanId;

    // Aggregate child input/output so it can be stamped onto the promoted root.
    if (this.agentTraceMode && !isPromotedRoot) {
      captureChildIO(span, agg);
    }

    // Mark trace as AI SDK trace if this span is an AI SDK span
    // (attributes are available at onEnd time, not onStart)
    if (isLikelyAISDKSpan(span)) {
      agg.isAISDKTrace = true;
    }

    // Only aggregate error state for AI SDK traces
    if (agg.isAISDKTrace) {
      const { error, message } = spanHasErrorSignal(span);
      if (error) {
        agg.hadError = true;
        if (agg.firstErrorMessage == null && message != null) {
          agg.firstErrorMessage = message;
        }
      }
    }

    // Promote the selected root (clear its parent) before rename/status, which
    // key off parentSpanId == null.
    if (isPromotedRoot) {
      promoteSpanToRoot(span);
    } else {
      maybeRenameRootSpan(span);
      if (agg.isAISDKTrace) {
        if (span.parentSpanId == null) {
          maybeSetRootStatus(span, agg);
        } else {
          maybeSetSpanOkStatus(span);
        }
      }
    }

    // Decrement active span count; the trace is complete when it hits zero.
    agg.activeSpans = Math.max(0, agg.activeSpans - 1);
    const complete = agg.activeSpans === 0;

    let toExport: ReadableSpan[];
    if (isPromotedRoot) {
      if (complete) {
        // Root is also the last span — finalize and export it now.
        this.finalizeRoot(span, agg);
        toExport = [span];
      } else {
        // Defer: hold the root until the rest of the trace has ended.
        agg.pendingRoot = span;
        toExport = [];
      }
    } else if (complete && agg.pendingRoot != null) {
      // Last span of a trace whose root was deferred — finalize and export both.
      this.finalizeRoot(agg.pendingRoot, agg);
      toExport = [span, agg.pendingRoot];
    } else {
      toExport = [span];
    }

    if (complete) {
      this.traceAggregates.delete(traceId);
    }
    return toExport;
  }

  /**
   * Applies the deferred finishing touches to a promoted root: ensure an AGENT
   * span kind, stamp the aggregated input/output, set the root name and status.
   */
  private finalizeRoot(root: ReadableSpan, agg: TraceAggregate): void {
    ensureAgentSpanKind(root);
    applyAggregatedIO(root, agg);
    maybeRenameRootSpan(root);
    if (agg.isAISDKTrace) {
      maybeSetRootStatus(root, agg);
    }
  }

  /**
   * Finalizes and returns any promoted roots still held for in-flight traces.
   * Called on shutdown/forceFlush so deferred roots are not lost if a trace has
   * not completed. Each returned root is removed from its aggregate.
   */
  drainPendingRoots(): ReadableSpan[] {
    const roots: ReadableSpan[] = [];
    for (const agg of this.traceAggregates.values()) {
      if (agg.pendingRoot != null) {
        this.finalizeRoot(agg.pendingRoot, agg);
        roots.push(agg.pendingRoot);
        agg.pendingRoot = undefined;
      }
    }
    return roots;
  }

  /**
   * Clear all tracked aggregates. Called during shutdown/forceFlush to prevent leaks.
   */
  clear(): void {
    this.traceAggregates.clear();
  }
}
