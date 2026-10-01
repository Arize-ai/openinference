---
"@arizeai/openinference-instrumentation-typesafe": minor
---

Record System One calls with the `decision.*` semantic conventions instead of `llm.*`: the model is identified by `decision.system`, `decision.provider`, `decision.model_name`, `decision.request.model_name`, and `decision.response.model_name`, and token usage is recorded as `decision.token_count.input` and `decision.token_count.output` (no total). The `llm.system`, `llm.provider`, `llm.*model_name`, and `llm.token_count.*` attributes are no longer emitted.
