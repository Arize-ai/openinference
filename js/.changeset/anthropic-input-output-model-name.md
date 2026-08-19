---
"@arizeai/openinference-semantic-conventions": minor
---

Add `llm.input_model_name` and `llm.output_model_name` semantic conventions, letting instrumentation record the model requested by the caller separately from the model that actually generated the response (e.g. provider-side classifier/fallback routing). `llm.model_name` keeps its existing meaning.
