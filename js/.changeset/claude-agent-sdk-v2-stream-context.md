---
"@arizeai/openinference-instrumentation-claude-agent-sdk": patch
---

Run the V2 session stream inside the ClaudeAgent.turn context so spans created while iterating it are parented on the turn span
