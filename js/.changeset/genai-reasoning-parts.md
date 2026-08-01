---
"@arizeai/openinference-genai": patch
---

Handle `reasoning` message parts as first-class content (`message_content.type = "reasoning"` with the reasoning text) and stop duplicating unrecognized parts into the flat `message.content` alongside `message.contents.*`.
