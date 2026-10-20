---
"@arizeai/openinference-instrumentation-openai": patch
---

Keep streamed parallel tool calls apart. The stream consumer recorded each tool-call delta at its position inside the chunk, so with parallel calls the ids, names and arguments of all calls were concatenated into tool call 0. Deltas are now matched to their call by `index`; for providers that omit `index`, a new id (or a name without an id) starts a new call.
