---
"@arizeai/openinference-instrumentation-anthropic": patch
---

Record every `tool_result` block of an input message as its own tool message. Previously only the last block survived on the span, so parallel tool-call replies lost all but one result. Tool messages are recorded ahead of any other content in the same message, matching the order the API requires.
