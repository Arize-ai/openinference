---
"@arizeai/openinference-semantic-conventions": minor
---

Add `llm.request.model_name` and `llm.response.model_name` semantic conventions, letting instrumentation record the model requested by the caller separately from the model that actually generated the response (e.g. provider-side classifier/fallback routing). `llm.model_name` keeps its existing meaning.
