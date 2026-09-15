---
"@arizeai/openinference-instrumentation-anthropic": patch
---

Capture prompt caching token counts as `llm.token_count.prompt_details.cache_write` and `llm.token_count.prompt_details.cache_read` for streaming and non-streaming `messages.create` calls. Anthropic's `input_tokens` excludes cached tokens, so `llm.token_count.prompt` and `llm.token_count.total` now include the cache write and read counts. Zero cache counts are omitted, and streaming usage is merged field by field across `message_start`, the server-side fallback hop and `message_delta`, so streaming and non-streaming spans report the same counts.
