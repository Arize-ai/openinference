---
"@arizeai/openinference-instrumentation-openai-agents": minor
"@arizeai/openinference-instrumentation-openai": minor
---

Capture `llm.token_count.prompt_details.cache_write` from OpenAI prompt cache usage (`cache_write_tokens`) on both the Chat Completions and Responses APIs. Chat Completions streams now record token usage from the final `stream_options.include_usage` chunk.
