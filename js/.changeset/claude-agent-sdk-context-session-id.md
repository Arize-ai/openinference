---
"@arizeai/openinference-instrumentation-claude-agent-sdk": patch
---

A session id supplied through OpenInference context (`setSession`) is no longer overwritten by the SDK's `session_id` on AGENT spans for `query()`, `unstable_v2_prompt()`, and V2 session turns. The SDK id still fills `session.id` when the caller has not set one.
