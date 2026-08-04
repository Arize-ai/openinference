import { type Context, diag } from "@opentelemetry/api";
import type { Span } from "@opentelemetry/sdk-trace-base";

import { getAttributesFromContext, withSafety } from "@arizeai/openinference-core";

/**
 * Copies OpenInference context attributes onto a span at start time.
 *
 * The Vercel AI SDK creates its own spans, so unlike the OpenInference instrumentors
 * (which build spans through an {@link https://github.com/Arize-ai/openinference OITracer})
 * this processor never reads the OpenInference context. That means values set with the
 * `@arizeai/openinference-core` helpers — `setSession`, `setUser`, `setMetadata`,
 * `setTags` — are dropped: a `session.id` set via
 *
 * ```typescript
 * context.with(setSession(context.active(), { sessionId }), () => streamText({ ... }))
 * ```
 *
 * would never reach the exported AI spans. {@link reparentOrphanedSpan} makes this worse:
 * once the HTTP/server span that carried the session is filtered out and the AI span is
 * re-rooted, there is nothing left holding the session id.
 *
 * When `propagateContextAttributes` is enabled this reads every OpenInference attribute
 * (`session.id`, `user.id`, `metadata.*`, `tag.tags`, …) from the start-time context and
 * writes it directly onto the span, so the values survive reparenting and export and
 * traces group into sessions in Arize / Phoenix. Setting them at `onStart` means children
 * started in the same context inherit them too.
 *
 * Wrapped in {@link withSafety} so a malformed context can never break the span pipeline.
 *
 * @param span - the span being started
 * @param parentContext - the context the span was started in
 */
export const propagateContextAttributesToSpan = withSafety({
  fn: (span: Span, parentContext: Context): void => {
    const attributes = getAttributesFromContext(parentContext);
    if (Object.keys(attributes).length > 0) {
      span.setAttributes(attributes);
    }
  },
  onError: (error) => {
    diag.warn(`Unable to propagate OpenInference context attributes to span: ${error}`);
  },
});
