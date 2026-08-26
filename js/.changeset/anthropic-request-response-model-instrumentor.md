---
"@arizeai/openinference-instrumentation-anthropic": minor
---

Capture `llm.request.model_name` and `llm.response.model_name` for streaming and non-streaming stable and beta `messages.create` calls, alongside the existing `llm.model_name`. Also update streamed response model attributes at server-side fallback boundaries so they identify the model that actually served the response.
