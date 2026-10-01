---
"@arizeai/openinference-instrumentation-typesafe": minor
---

Record `TypeSafeClient.systemOne` calls as `DECISION` spans instead of `LLM` spans. System One scores or selects among the candidate options supplied in the request rather than generating free-form text, which is what the new `DECISION` span kind describes. The `llm.*` attributes (provider, system, model names, invocation parameters, token counts) are unchanged.
