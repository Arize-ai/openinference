---
"@arizeai/openinference-vercel": minor
---

Add OpenInference decision model telemetry for AI SDK v7 `experimental_decide` calls. Decision model calls become `DECISION` spans with `decision.*` model, provider, system, and token count attributes, and the new `enrichSpanWithOpenInference` callback (pass it as `OpenTelemetry`'s `enrichSpan` option; requires `ai` 7.0.128 and `@ai-sdk/otel` 1.0.128 or later) classifies the outer decide operation as a `CHAIN` so that model identity and usage are not duplicated. The GenAI model and usage attributes (`gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.*`) are removed from decision spans after mapping so that GenAI converters do not re-label the decision model as an LLM.
