---
"@arizeai/openinference-vercel": minor
---

Add a `propagateContextAttributes` option to `OpenInferenceSimpleSpanProcessor` and `OpenInferenceBatchSpanProcessor`. The Vercel AI SDK creates its own spans, so unlike the OpenInference instrumentors (which build spans through an `OITracer`) this processor never reads the OpenInference context — meaning values set with the `@arizeai/openinference-core` helpers (`setSession`, `setUser`, `setMetadata`, `setTags`) never reach the exported AI spans. For example, a `session.id` set via `context.with(setSession(context.active(), { sessionId }), () => streamText(...))` would be dropped, and `reparentOrphanedSpans` makes this worse: once the HTTP/server span that carried it is filtered out and the AI span is re-rooted, nothing is left holding the session id.

When enabled, every OpenInference attribute present on the start-time context (`session.id`, `user.id`, `metadata.*`, `tag.tags`, …) is written directly onto the span at `onStart`, so the values survive reparenting and export and traces group into sessions in Arize / Phoenix. Setting them at start time means children started in the same context inherit them too. The read is wrapped in `withSafety`, so a malformed context can never break the span pipeline.

Defaults to `true`; set `propagateContextAttributes: false` to opt out. Packages that extend these processors inherit the option.
