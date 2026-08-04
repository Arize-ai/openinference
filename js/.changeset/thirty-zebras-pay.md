---
"@arizeai/openinference-instrumentation-anthropic": patch
---

Anthropic instrumentation now captures Claude extended thinking content in OpenInference message contents. Anthropic thinking blocks are recorded as reasoning content with their text and signature, while redacted_thinking blocks are recorded as reasoning content with their redacted data payload. This works for both streaming and non-streaming Messages responses, preserves content block ordering.
