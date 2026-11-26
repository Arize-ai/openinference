---
"@arizeai/openinference-instrumentation-claude-agent-sdk": patch
---

Return the SDK's own `Query` object from the instrumented `query()` so control methods such as `interrupt()` and `setPermissionMode()` keep working. The wrapper now calls the SDK at `query()` time, as the unwrapped SDK does, instead of at first iteration, and traces iteration driven through `next()`/`return()`/`throw()` as well as `for await`.
