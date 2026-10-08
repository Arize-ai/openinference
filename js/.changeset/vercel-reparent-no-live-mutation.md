---
"@arizeai/openinference-vercel": patch
---

`reparentOrphanedSpans` no longer mutates the caller's live span at `onStart`. The re-rooting is applied only to the exported span, so host runtimes that still reference the original parent (e.g. Vercel `eve`) no longer log "Operation attempted on ended Span" warnings.
