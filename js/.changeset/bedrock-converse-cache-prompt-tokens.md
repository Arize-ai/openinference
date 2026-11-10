---
"@arizeai/openinference-instrumentation-bedrock": patch
---

Include cache read and cache write tokens in `llm.token_count.prompt` for Converse and ConverseStream, so that prompt plus completion matches `llm.token_count.total` when prompt caching is on. Also record `llm.token_count.prompt_details.cache_read` and `llm.token_count.prompt_details.cache_write`.
