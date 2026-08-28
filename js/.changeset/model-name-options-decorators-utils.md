---
"@arizeai/openinference-core": minor
---

Add optional `requestModelName` and `responseModelName` options to `getLLMAttributes`, emitting the `llm.request.model_name` and `llm.response.model_name` semantic conventions. Per the spec, `llm.model_name` is mirrored from `responseModelName ?? requestModelName` when `modelName` is not passed explicitly. Usable with `withSpan` and the `@observe` decorator via `processInput` and `processOutput`
