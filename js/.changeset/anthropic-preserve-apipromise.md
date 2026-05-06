---
"@arizeai/openinference-instrumentation-anthropic": patch
---

fix: preserve `APIPromise` return type when instrumenting `Messages.prototype.create`. The previous `.then(...).catch(...)` wrapping collapsed the SDK's `APIPromise` to a plain `Promise` and stripped helpers like `.withResponse()` / `.asResponse()`, breaking `messages.stream(...)` (which internally calls `messages.create({ stream: true }).withResponse()`). The wrapper now uses an `invokeMaybeAPIPromise` helper that calls `APIPromise._thenUnwrap(...)`, mirroring the existing approach in `@arizeai/openinference-instrumentation-openai`.
