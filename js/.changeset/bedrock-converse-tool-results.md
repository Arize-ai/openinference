---
"@arizeai/openinference-instrumentation-bedrock": patch
---

Record each `toolResult` block of a Converse input message as its own tool message, with its `tool_call_id` and its text and JSON output as `message.content`. Previously the message kept only the last `toolUseId` and none of the tool output, so parallel tool results could not be linked or read.
