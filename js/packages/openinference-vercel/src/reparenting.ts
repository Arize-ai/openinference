import { type AttributeValue, type Context, trace } from "@opentelemetry/api";
import type { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { isLikelyAISDKSpan } from "./typeUtils.js";

/**
 * Decides whether an AI span would be orphaned by span filtering, and so should be
 * re-rooted (detached from its non-AI parent) in the exported trace.
 *
 * When a span filter drops non-OpenInference spans (e.g. `isOpenInferenceSpan`), the
 * highest-level AI span (e.g. `ai.generateText`, `ai.streamText`, or a framework "turn"
 * wrapper) is often parented under a non-AI span — such as the HTTP/server span Next.js
 * parents everything under. That parent is filtered out, leaving the AI span pointing at
 * a parent that was never exported, so backends may not be able to render the trace correctly.
 *
 * This is a pure predicate evaluated at `onStart`: it does NOT mutate the span. The
 * re-rooting is applied later, only to a read-only export view built at `onEnd` (see
 * {@link createReparentedSpanView}). Clearing the parent on the live span at `onStart`
 * instead severs a linkage a host runtime may still reference — e.g. Vercel's `eve`
 * durable-workflow runtime — driving it into operations on already-ended spans and flooding
 * logs with "Operation attempted on ended Span" warnings (issue #3292). The check is stateless:
 * the parent span is read directly from the start-time `parentContext`, so no per-trace
 * bookkeeping is needed. Spans whose parent is itself an AI span are left attached, keeping
 * the AI subtree intact, and multiple sibling AI spans are each re-rooted independently.
 *
 * @param span - the span being started
 * @param parentContext - the context the span was started in (carries the parent span)
 * @returns true if the span should be re-rooted in the exported trace
 */
export const shouldReparentSpan = (span: Span, parentContext: Context): boolean => {
  // Re-rooting is decided at onStart, but the OpenInference span kind the export filter
  // checks isn't assigned until onEnd — so we can't ask "will this be exported?" yet. Use
  // the start-time AI/GenAI heuristic as a proxy for a span we care about. A false positive
  // is harmless: if it isn't actually relevant, the filter drops it at export anyway.
  if (!isLikelyAISDKSpan(span)) return false;

  const parentSpan = trace.getSpan(parentContext);
  // No parent — already a root.
  if (parentSpan == null) return false;

  // The parent must be inspectable to judge it. Across an async/durable boundary (e.g. a
  // framework's workflow steps) the parent arrives as a non-recording span — only a
  // SpanContext, with no `attributes` — even though its spanId is correct and the parent is
  // itself exported. "Can't inspect" is NOT "non-AI": treating it as non-AI re-roots the child
  // off an exported AI parent (orphaning it). When the parent isn't inspectable, leave the
  // child attached; the parentSpanId link stays valid if the parent is exported.
  const parentAttributes = (parentSpan as unknown as { attributes?: unknown }).attributes;
  if (parentAttributes == null) return false;

  // The parent is inspectable and also looks like an AI span (so it will be exported too and
  // the link is fine). Only re-root when the parent is a real, inspectable, non-AI span — i.e.
  // likely to be filtered out, which is what would orphan this span.
  if (isLikelyAISDKSpan(parentSpan as unknown as Span)) return false;

  return true;
};

/**
 * Builds a read-only, re-rooted view of a span for export.
 *
 * The returned {@link ReadableSpan} behaves exactly like `span` except its parent linkage
 * reads as detached (`parentSpanId` / `parentSpanContext` are `undefined`), so it is exported
 * as a trace root while the caller's live span keeps its real parent and is never mutated —
 * this is what avoids driving a host runtime into post-end span operations (issue #3292).
 *
 * The processor runs its export-time conversion (attribute mapping, root rename, root status,
 * AGENT promotion) against this view, so the steps that key off "is a root" observe the
 * re-rooted parent without touching the original span. `attributes` and `events` are shallow
 * copies so those writes land on the view, not on the live span; `name`/`status` reassignments
 * are captured on the view by the proxy's set trap. Every other field (including methods and
 * computed getters like `spanContext()` and `duration`) is forwarded to the live span.
 *
 * @param span - the span ending; the source the view mirrors
 * @returns a re-rooted, read-only view suitable for export
 */
export const createReparentedSpanView = (span: ReadableSpan): ReadableSpan => {
  // A null-prototype store so `prop in overrides` only ever matches keys we set here.
  const overrides: Record<PropertyKey, unknown> = Object.create(null);
  overrides.parentSpanId = undefined;
  overrides.parentSpanContext = undefined;
  overrides.attributes = { ...span.attributes };
  overrides.events = [...span.events];

  return new Proxy(span, {
    get(target, prop) {
      if (prop in overrides) return overrides[prop];
      const value = Reflect.get(target, prop, target);
      return typeof value === "function" ? value.bind(target) : value;
    },
    set(_target, prop, value) {
      // Capture reassignments (e.g. the root rename / root status writes) on the view so the
      // live span is left untouched.
      overrides[prop] = value;
      return true;
    },
  });
};

/**
 * Gives a kind-less root AI span a fallback `openinference.span.kind` so it survives an
 * OpenInference span filter and serves as the exported trace root.
 *
 * Some framework "wrapper" spans (e.g. a per-turn span an agent framework emits on top of
 * the Vercel AI SDK) carry an `ai.*` operation name the kind map doesn't recognize, so
 * attribute conversion leaves them without a span kind — which means a filter would drop
 * them even after re-rooting makes them a root, re-orphaning their AI children. This tags
 * such a span `AGENT` so it is recognized and kept.
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
