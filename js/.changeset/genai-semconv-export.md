---
"@arizeai/openinference-core": minor
---

feat(core): add experimental `enableGenAISemconv` trace config option and `OPENINFERENCE_ENABLE_GENAI_SEMCONV` environment variable. When enabled, spans created through `OITracer` also emit OpenTelemetry GenAI semantic convention (`gen_ai.*`) attributes, derived at span end from the span's masked OpenInference attributes. The conversion is also exported as `getGenAIAttributes`.
