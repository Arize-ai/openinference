---
"@arizeai/openinference-vercel": patch
---

Classify Vercel eve's `agent.step`, `agent.action`, and `agent.approval` control-flow spans as CHAIN instead of LLM (an `agent.action` for a subagent or remote-agent call stays AGENT). They carry `gen_ai.*` context but no `gen_ai.operation.name`, so they fell through to the GenAI converter's LLM default and inflated the LLM span count well beyond the number of model calls. A `session.id` already on a span (for example from `setSession` context) now takes precedence over the one derived from `gen_ai.conversation.id`.
