---
"@arizeai/openinference-instrumentation-claude-agent-sdk": patch
---

`query()` now returns the SDK's `Query` object instead of a plain async iterable.

The wrapper built a new `{ [Symbol.asyncIterator] }` object, so every control method the SDK exposes on `Query` — `interrupt`, `setPermissionMode`, `setModel`, `setMaxThinkingTokens`, `setMcpPermissionModeOverride`, `applyFlagSettings`, `reinitialize`, `initializationResult`, and `next`/`return`/`throw` — was `undefined` on the instrumented result. Enabling tracing therefore removed the documented way to stop a running query.

The wrapped function also called the SDK lazily inside `[Symbol.asyncIterator]`, shifting Claude Code subprocess start until the first iteration. It now calls `query()` eagerly and returns a proxy over the real `Query` that delegates every other member to the SDK object, preserving the unwrapped behaviour with tracing as the only difference.
