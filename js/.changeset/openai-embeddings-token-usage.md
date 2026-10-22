---
"@arizeai/openinference-instrumentation-openai": patch
---

Record token usage on `embeddings.create` spans: `llm.token_count.prompt` and `llm.token_count.total` from the response's `usage`, as the Python OpenAI instrumentor does. Previously embedding spans carried no token counts, so embedding usage was missing from traces.
