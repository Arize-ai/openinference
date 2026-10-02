---
"@arizeai/openinference-instrumentation-claude-agent-sdk": patch
---

Count prompt-cache tokens on the AGENT span. `llm.token_count.prompt` and `llm.token_count.total` now include `cache_read_input_tokens` and `cache_creation_input_tokens`, and `llm.token_count.prompt_details.cache_read` and `cache_write` are set, matching the Python instrumentor. Previously the prompt count was `input_tokens` alone, which excludes the cache, so a typical Claude Agent SDK run reported single-digit prompt counts.
