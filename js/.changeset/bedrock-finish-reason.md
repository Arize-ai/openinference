---
"@arizeai/openinference-instrumentation-bedrock": minor
---

Record provider-native stop reasons as `llm.finish_reason` for Converse, InvokeModel, and streaming responses. Preserve the existing Converse `llm.stop_reason` attribute and omit the finish reason when unavailable.
