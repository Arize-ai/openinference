---
"@arizeai/openinference-vercel": patch
---

Fix `reparentOrphanedSpans` mutating the caller's live span. Re-rooting an orphaned AI span previously cleared `parentSpanId` / `parentSpanContext` on the live `Span` object at `onStart`. The Vercel AI SDK tolerates this, but host runtimes that manage span lifecycle around that parent — notably Vercel's `eve` durable-workflow agent framework — are then driven into operating on already-ended spans, producing a flood of `Operation attempted on ended Span` warnings (150+ per turn) on the OpenTelemetry SDK.

The processor now records re-rooting intent at `onStart` (without touching the span) and applies the parent clearing only to a read-only view exported at `onEnd`, leaving the caller's live span untouched. Exported trace output is unchanged: orphaned AI spans are still re-rooted, kind-less AI wrappers (e.g. `ai.eve.turn`) are still promoted to `AGENT`, and the option still defaults to `false`.
