---
"@arizeai/openinference-core": patch
---

Fix `withSpan` to properly handle synchronous errors, preserve `this` binding on the wrapped function, defer default tracer resolution until invocation time, and clarify the agent-facing docs/examples
