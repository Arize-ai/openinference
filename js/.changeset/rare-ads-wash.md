---
"@arizeai/openinference-tanstack-ai": major
"@arizeai/openinference-genai": minor
---

feat: Wrap native tanstack/otel package within openinference-tanstack

Refactor TanStack AI instrumentation to wrap TanStack's native OTEL middleware and convert its GenAI spans to OpenInference attributes.
This decouples OpenInference from TanStack AI's request, model-turn, streaming, tool-call, finish, error, and abort lifecycle management while preserving the `openInferenceMiddleware()` user API.

`@arizeai/openinference-tanstack-ai` now requires `@tanstack/ai >=0.15.0` for native OTEL middleware support. 
`@arizeai/openinference-genai` now exposes reusable span-level GenAI-to-OpenInference conversion utilities used by the TanStack adapter and available to other GenAI telemetry integrations.