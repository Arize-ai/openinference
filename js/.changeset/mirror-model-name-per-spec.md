---
"@arizeai/openinference-core": patch
---

Mirror `llm.model_name` from `responseModelName ?? requestModelName` in `getLLMAttributes` when `modelName` is not passed explicitly, as the spec requires; document composing the options with `withSpan` and the `@observe` decorator via `processInput` and `processOutput`
