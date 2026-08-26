---
"@arizeai/openinference-instrumentation-anthropic": minor
---

Instrument `beta.messages.create` and capture `llm.request.model_name` and `llm.response.model_name` for streaming and non-streaming stable and beta `messages.create` calls, alongside the existing `llm.model_name`. Server-side fallback is now traced end to end: the response model attributes are updated at fallback boundaries so they identify the model that actually served the response, the `fallback` content block is recorded as message content (with the declining model and refusal category) instead of leaving a hole in `message_contents`, token counts are taken from the serving attempt rather than mixed across models, and `llm.finish_reason` is recorded so classifier refusals (`stop_reason: "refusal"`) are visible on the span.
