---
"@arizeai/openinference-core": minor
---

Add optional `requestModelName` and `responseModelName` options to `getLLMAttributes`, emitting the `llm.request.model_name` and `llm.response.model_name` semantic conventions while keeping `llm.model_name` resolved for backwards-compatible consumers; usable with `withSpan` and the `@observe` decorator via `attributes` and `processOutput`
