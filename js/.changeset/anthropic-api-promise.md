---
"@arizeai/openinference-instrumentation-anthropic": patch
---

Preserve `APIPromise` helpers (`withResponse()` / `asResponse()`) on the patched `messages.create`, fixing `client.messages.stream()` throwing `create(...).withResponse is not a function` when instrumented.
