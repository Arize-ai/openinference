---
"@arizeai/openinference-vercel": patch
---

Read the parent span from `parentSpanContext` (OpenTelemetry JS SDK 2.x) with a fallback to `parentSpanId` (SDK 1.x) when deciding whether a span is the trace root. Under SDK 2.x every span was treated as a root, so child AI SDK spans were renamed to their `operation.name`, a later successful child span inherited an earlier sibling's ERROR status, and nested kind-less AI spans were promoted to AGENT when `reparentOrphanedSpans` is enabled.
