---
"@arizeai/openinference-instrumentation-openai": patch
"@arizeai/openinference-instrumentation-openai-agents": patch
---

fix: leave the base64 `result` of `image_generation_call` items out of `output.value` on Responses spans and OpenAI Agents response spans. The image stays on `output.images.*`, where `base64ImageMaxLength` applies, as the Images API path already does for `b64_json`. This changes `output.value` for Responses users of the `image_generation` tool: the `result` field is dropped rather than replaced with a placeholder.
