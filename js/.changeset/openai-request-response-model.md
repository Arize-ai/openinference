---
"@arizeai/openinference-instrumentation-openai": minor
---

feat(openai): record `llm.request.model_name` and `llm.response.model_name` on chat completion, completion, Responses and image spans. `llm.model_name` is unchanged: it still holds the requested model until a response arrives and the served model afterwards. Streamed chat completions now also update `llm.model_name` to the served model, as non-streamed ones already did.
