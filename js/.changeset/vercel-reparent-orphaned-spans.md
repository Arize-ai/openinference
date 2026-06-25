---
"@arizeai/openinference-vercel": minor
---

Add an opt-in `reparentOrphanedSpans` option to `OpenInferenceSimpleSpanProcessor` and `OpenInferenceBatchSpanProcessor`. When a span filter drops non-OpenInference spans (e.g. `isOpenInferenceSpan`), the highest-level AI span (such as `ai.generateText`/`ai.streamText` parented under the HTTP/server span Next.js parents everything under) is otherwise left orphaned — pointing at a parent that was never exported, so backends may not be able to render the trace correctly. With this enabled, any AI span whose direct parent is a non-AI span is detached (re-rooted) so it becomes a trace root. The check is stateless (the parent is read from the start-time context). Handles multiple sibling AI spans per trace; AI spans nested under an AI parent are left intact.

If the re-rooted span is an `ai.*` framework wrapper that the package doesn't map to a span kind (e.g. a per-turn span an agent framework emits on top of the AI SDK), it would otherwise be kind-less and dropped by the filter; such a root is tagged `openinference.span.kind = AGENT` so it is kept. This is matched by shape (an unrecognized AI-like root), not by any specific span name.

Defaults to `false`, so existing behavior is unchanged. It is intended for use alongside a filter that drops non-AI parent spans. Packages that extend these processors inherit the option.
