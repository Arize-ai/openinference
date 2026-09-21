---
"@arizeai/openinference-instrumentation-anthropic": patch
---

Record every `tool_result` block of an input message as its own tool message. Previously only the last block survived on the span, so parallel tool-call replies lost all but one result.
