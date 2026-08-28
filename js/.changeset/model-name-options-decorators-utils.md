---
"@arizeai/openinference-core": minor
---

Add optional `requestModelName` and `responseModelName` options to `getLLMAttributes`, emitting the `llm.request.model_name` and `llm.response.model_name` semantic conventions; usable with `withSpan` and the `@observe` decorator via `attributes` and `processOutput`
