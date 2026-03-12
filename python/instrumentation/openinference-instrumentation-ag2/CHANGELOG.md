## [0.1.0] — 2026-03-11

### Added
- `AG2Instrumentor` with `BaseInstrumentor` + `wrapt` + `OITracer` pattern
- Wrappers for: `initiate_chat`, `generate_reply`, `execute_function`, `GroupChatManager.run_chat`, `initiate_swarm_chat`, `ReasoningAgent.generate_response`, `initiate_chats`
- Phoenix graph view compatibility via `graph.node.id` attributes
- AG2-specific span attributes in `ag2.*` namespace
- Targets `ag2>=0.11.0` (community fork); distinct from `openinference-instrumentation-autogen-agentchat` (Microsoft fork)
