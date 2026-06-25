---
"@arizeai/openinference-vercel": minor
---

Add an `agentTraceMode` option to `OpenInferenceSimpleSpanProcessor` and `OpenInferenceBatchSpanProcessor` (default `false`). When enabled, the processor produces a clean agent trace tree for the Vercel AI SDK and frameworks built on it: it promotes the first `ai.*` span in each trace (including framework wrappers such as Eve's `ai.eve.turn`) to the trace root, labels that root `AGENT` when it has no span kind, propagates `session.id` from the active context, and stamps the trace's earliest input / latest output onto the root when it has none of its own. Also adds a `preProcessSpan` hook (and exported `PreProcessSpan` type) for framework-specific attribute enrichment before OpenInference conversion runs.
