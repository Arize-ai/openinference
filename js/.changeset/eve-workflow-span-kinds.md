---
"@arizeai/openinference-vercel": patch
---

Keep classifying Vercel eve's `agent.step` and `agent.approval` spans as CHAIN on eve 0.76 and later. eve 0.76 sets `operation.name` and `gen_ai.operation.name` on these spans to `workflow` and moves the span name to `resource.name`, so they no longer matched the eve span-kind map and fell through to the GenAI converter's LLM default, which doubled a turn's LLM span count. The processor now also matches `resource.name` against that map.
