---
"@arizeai/openinference-core": minor
---

Add reasoning content support to `getLLMAttributes`. The `Message.contents` array now accepts a `{ type: "reasoning", text?, signature?, data?, encryptedContent? }` entry, and `ToolCall` accepts an optional `reasoningSignature`. These emit:

- `llm.{input,output}_messages.*.message.contents.*.message_content.type = "reasoning"`
- `llm.{input,output}_messages.*.message.contents.*.message_content.text`
- `llm.{input,output}_messages.*.message.contents.*.message_content.signature`
- `llm.{input,output}_messages.*.message.contents.*.message_content.data`
- `llm.{input,output}_messages.*.message.contents.*.message_content.encrypted_content`
- `llm.{input,output}_messages.*.message.tool_calls.*.tool_call.reasoning_signature`

No `message_content.id` is emitted for reasoning entries. Reasoning `text` is human-readable and is redacted by `hideInputText` / `hideOutputText` like other `message_content.text`. The opaque echo-token fields (`signature`, `data`, `encrypted_content`, `tool_call.reasoning_signature`) are removed by `hideInputMessages` / `hideOutputMessages` but are intentionally preserved through `hideInputText` / `hideOutputText`.
