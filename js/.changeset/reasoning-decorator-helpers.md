---
"@arizeai/openinference-core": minor
---

Add reasoning content support to `getLLMAttributes`. The `Message.contents` array now accepts a `{ type: "reasoning", signature?, data?, encryptedContent? }` entry, and `ToolCall` accepts an optional `reasoningSignature`. These emit:

- `llm.{input,output}_messages.*.message.contents.*.message_content.type = "reasoning"`
- `llm.{input,output}_messages.*.message.contents.*.message_content.signature`
- `llm.{input,output}_messages.*.message.contents.*.message_content.data`
- `llm.{input,output}_messages.*.message.contents.*.message_content.encrypted_content`
- `llm.{input,output}_messages.*.message.tool_calls.*.tool_call.reasoning_signature`

No `message_content.id` is emitted for reasoning entries. The new opaque echo-token fields are removed by `hideInputMessages`/`hideOutputMessages` but are preserved through `hideInputText`/`hideOutputText`, which only target user-visible text.
