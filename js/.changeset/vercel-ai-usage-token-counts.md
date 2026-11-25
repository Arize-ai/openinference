---
"@arizeai/openinference-vercel": patch
---

Map `ai.usage.totalTokens`, `ai.usage.reasoningTokens` and `ai.usage.cachedInputTokens` to `llm.token_count.total`, `llm.token_count.completion_details.reasoning` and `llm.token_count.prompt_details.cache_read`. `gen_ai.usage.*` values still take precedence.
