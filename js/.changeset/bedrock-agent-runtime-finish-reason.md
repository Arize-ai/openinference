---
"@arizeai/openinference-instrumentation-bedrock-agent-runtime": minor
---

Record provider-native stop reasons from agent model trace responses as `llm.finish_reason` on model invocation spans. Preserve the original values and omit the attribute when the reason is unavailable or invalid.
