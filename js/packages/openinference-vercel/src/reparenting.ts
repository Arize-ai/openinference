import { type AttributeValue, type Context, type SpanContext, trace } from "@opentelemetry/api";
import type { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { isLikelyAISDKSpan } from "./typeUtils";

/**
 * Reparents an AI span that would be orphaned by span filtering.
 *
 * When a span filter drops non-OpenInference spans (e.g. `isOpenInferenceSpan`), the
 * highest-level AI span (e.g. `ai.generateText`, `ai.streamText`, or a framework "turn"
 * wrapper) is often parented under a non-AI span — such as the HTTP/server span Next.js
 * parents everything under. That parent is filtered out, leaving the AI span pointing at
 * a parent that was never exported, so backends may not be able to render the trace correctly.
 *
 * This detaches such a span (clears its `parentSpanId` / `parentSpanContext`) so it
 * becomes a trace root. It is fully stateless: the parent span is read directly from the
 * start-time `parentContext`, so no per-trace bookkeeping is needed. Spans whose parent
 * is itself an AI span are left untouched, keeping the AI subtree intact, and multiple
 * sibling AI spans are each re-rooted independently.
 *
 * Runs at `onStart`, before the span is mutated/exported.
 *
 * @param span - the span being started
 * @param parentContext - the context the span was started in (carries the parent span)
 */
export const reparentOrphanedSpan = (span: Span, parentContext: Context): void => {
  // Re-rooting must happen at onStart, but the OpenInference span kind the export filter
  // checks isn't assigned until onEnd — so we can't ask "will this be exported?" yet. Use
  // the start-time AI/GenAI heuristic as a proxy for a span we care about. A false positive
  // is harmless: if it isn't actually relevant, the filter drops it at export anyway.
  if (!isLikelyAISDKSpan(span)) return;

  const parentSpan = trace.getSpan(parentContext);
  // No parent (already a root), or the parent also looks like an AI span (so it will be
  // exported too and the link is fine). Only re-root when the parent looks non-AI — i.e.
  // likely to be filtered out, which is what would orphan this span.
  if (parentSpan == null || isLikelyAISDKSpan(parentSpan as unknown as Span)) return;

  // Detach from the non-AI parent so this span becomes a trace root. The parent fields are
  // typed readonly on ReadableSpan, but the live Span object at onStart is mutable — re-type
  // it as writable to clear both the 1.x (`parentSpanId`) and 2.x (`parentSpanContext`) shapes.
  const writableSpan = span as unknown as {
    parentSpanId?: string;
    parentSpanContext?: SpanContext;
  };
  writableSpan.parentSpanId = undefined;
  writableSpan.parentSpanContext = undefined;
};

/**
 * Gives a kind-less root AI span a fallback `openinference.span.kind` so it survives an
 * OpenInference span filter and serves as the exported trace root.
 *
 * Some framework "wrapper" spans (e.g. a per-turn span an agent framework emits on top of
 * the Vercel AI SDK) carry an `ai.*` operation name the kind map doesn't recognize, so
 * attribute conversion leaves them without a span kind — which means a filter would drop
 * them even after {@link reparentOrphanedSpan} makes them a root, re-orphaning their AI
 * children. This tags such a span `AGENT` so it is recognized and kept.
 *
 * `AGENT` is the right default precisely because only AI-like roots are ever promoted:
 * non-AI spans are filtered out, never promoted, so a promoted span is the top-level AI
 * wrapper of its trace. This stays name-agnostic — it keys off span shape (a root, AI-like
 * span the kind map left without a kind), not any framework name. A framework that wants a
 * different kind for its own wrapper can set it before this runs (this only fires when the
 * kind is still unset).
 *
 * Recognized spans (the kind map assigned a kind) and nested spans are untouched. Runs at
 * onEnd, after attribute conversion and before the export filter.
 */
export const promoteReparentedRoot = (span: ReadableSpan): void => {
  if (span.parentSpanId != null) return;
  if (span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] != null) return;
  if (!isLikelyAISDKSpan(span)) return;

  // ReadableSpan attributes are typed readonly; runtime Span objects are mutable.
  (span.attributes as Record<string, AttributeValue>)[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
    OpenInferenceSpanKind.AGENT;
};
