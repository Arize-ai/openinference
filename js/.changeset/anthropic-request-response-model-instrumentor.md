---
"@arizeai/openinference-instrumentation-anthropic": minor
---

Capture `llm.request.model_name` and `llm.response.model_name` for streaming and non-streaming `messages.create` calls, alongside the existing `llm.model_name`. Also fixes a bug where `llm.model_name` was never updated to reflect the model that actually served a streamed response, staying stuck at the requested model.
