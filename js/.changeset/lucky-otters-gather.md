---
"@arizeai/openinference-semantic-conventions": minor
---

Add `META`, `ZAI`, and `MINIMAX` to the `LLMProvider` enum, giving Meta AI (`https://api.meta.ai/v1`, `muse-spark-*` models), Z.ai (`https://api.z.ai/api/paas/v4`, GLM models), and MiniMax (`https://api.minimax.io/v1`) well-known `llm.provider` values instead of leaving each to a custom string. Mirrors the same additions in the Python, Java, and Go semantic conventions and in the spec's well-known value table.
