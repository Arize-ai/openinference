---
"@arizeai/openinference-vercel": patch
---

Classify Vercel eve's `agent.step` and `agent.action` control-flow spans as CHAIN instead of LLM. They carry `gen_ai.*` context but no `gen_ai.operation.name`, so they fell through to the GenAI converter's LLM default and each turn appeared to make twice as many model calls. A `session.id` already on a span (for example from `setSession` context) now takes precedence over the one derived from `gen_ai.conversation.id`.
