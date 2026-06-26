import { type AttributeValue, type Context, trace } from "@opentelemetry/api";
import type { ReadableSpan, Span } from "@opentelemetry/sdk-trace-base";

import {
  OpenInferenceSpanKind,
  SemanticConventions,
} from "@arizeai/openinference-semantic-conventions";

import { isLikelyAISDKSpan } from "./typeUtils";

/**
 * Decides whether an AI span would be orphaned by span filtering and should be re-rooted.
 *
 * When a span filter drops non-OpenInference spans (e.g. `isOpenInferenceSpan`), the
 * highest-level AI span (e.g. `ai.generateText`, `ai.streamText`, or a framework "turn"
 * wrapper) is often parented under a non-AI span — such as the HTTP/server span Next.js
 * parents everything under. That parent is filtered out, leaving the AI span pointing at
 * a parent that was never exported, so backends may not be able to render the trace correctly.
 *
 * This is a pure, stateless predicate: the parent span is read directly from the start-time
 * `parentContext`, so no per-trace bookkeeping is needed. It does NOT mutate the span — the
 * actual re-rooting is applied to the exported view at onEnd (see {@link createReparentedSpan}),
 * leaving the caller's live span object untouched. Spans whose parent is itself an AI span
 * return `false`, keeping the AI subtree intact; multiple sibling AI spans are each judged
 * independently.
 *
 * Runs at `onStart`, where the parent context is available.
 *
 * @param span - the span being started
 * @param parentContext - the context the span was started in (carries the parent span)
 * @returns `true` if the span should be re-rooted on export
 */
export const shouldReparentOrphanedSpan = (span: Span, parentContext: Context): boolean => {
  // Re-rooting must be decided at onStart, but the OpenInference span kind the export filter
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
 * Returns a read-only view of `span` with its parent linkage cleared, so the exporter sees a
 * trace root. The underlying span is left untouched — re-rooting must not mutate the caller's
 * live `Span` object, because host runtimes that manage span lifecycle around that parent (e.g.
 * a durable-workflow agent framework) can be driven into operating on already-ended spans by an
 * in-place parent mutation, producing a flood of "Operation attempted on ended Span" warnings.
 *
 * The proxy forwards every read to the real span (including the `spanContext()` method and any
 * attribute/status/name updates applied at onEnd) and only overrides `parentSpanId` /
 * `parentSpanContext` to `undefined`, covering both the 1.x and 2.x ReadableSpan shapes.
 */
export const createReparentedSpan = (span: ReadableSpan): ReadableSpan =>
  new Proxy(span, {
    get(target, prop) {
      if (prop === "parentSpanId" || prop === "parentSpanContext") {
        return undefined;
      }
      // Read off the real span, binding methods (e.g. spanContext()) to it so they keep
      // working; using `target` as the getter receiver resolves getters against the real span.
      const value = Reflect.get(target, prop, target);
      return typeof value === "function"
        ? (value as (...args: unknown[]) => unknown).bind(target)
        : value;
    },
  });

/**
 * Gives a kind-less re-rooted AI span a fallback `openinference.span.kind` so it survives an
 * OpenInference span filter and serves as the exported trace root.
 *
 * Some framework "wrapper" spans (e.g. a per-turn span an agent framework emits on top of
 * the Vercel AI SDK) carry an `ai.*` operation name the kind map doesn't recognize, so
 * attribute conversion leaves them without a span kind — which means a filter would drop
 * them even after {@link shouldReparentOrphanedSpan} marks them for re-rooting, re-orphaning
 * their AI children. This tags such a span `AGENT` so it is recognized and kept.
 *
 * `AGENT` is the right default precisely because only AI-like roots are ever re-rooted:
 * non-AI spans are filtered out, never re-rooted, so a re-rooted span is the top-level AI
 * wrapper of its trace. This stays name-agnostic — it keys off span shape (an AI-like span the
 * kind map left without a kind), not any framework name. A framework that wants a different
 * kind for its own wrapper can set it before this runs (this only fires when the kind is still
 * unset).
 *
 * Setting an attribute (not a parent field) on the already-ended span is safe and does not
 * affect the host's span lifecycle. The caller only invokes this for spans it is re-rooting,
 * so there is no parent-id check here. Runs at onEnd, after attribute conversion and before the
 * export filter.
 */
export const promoteReparentedRoot = (span: ReadableSpan): void => {
  if (span.attributes[SemanticConventions.OPENINFERENCE_SPAN_KIND] != null) return;
  if (!isLikelyAISDKSpan(span)) return;

  // ReadableSpan attributes are typed readonly; runtime Span objects are mutable.
  (span.attributes as Record<string, AttributeValue>)[SemanticConventions.OPENINFERENCE_SPAN_KIND] =
    OpenInferenceSpanKind.AGENT;
};
