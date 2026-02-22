---
"@arizeai/openinference-instrumentation-claude-agent-sdk": patch
---

fix: improve type safety and robustness of hook injection

- Make mergeHooks generic to eliminate unsafe Record<string, unknown> casts
- Add asRecord helper for safe toolInput coercion
- Wrap hook callbacks in try-catch to prevent instrumentation from breaking the SDK
- Use hook_event_name discriminant narrowing instead of as casts
- Store and restore original functions in unpatch() to prevent double-wrapping
