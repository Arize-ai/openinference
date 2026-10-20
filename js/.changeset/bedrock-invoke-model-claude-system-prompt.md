---
"@arizeai/openinference-instrumentation-bedrock": patch
---

Record the top-level `system` prompt of Anthropic `InvokeModel` requests as the first input message (a string or an array of text blocks), so the conversation's messages start at index 1 when a system prompt is set.
