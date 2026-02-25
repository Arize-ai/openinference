# @arizeai/openinference-instrumentation-claude-agent-sdk

## 0.2.0

### Minor Changes

- cacb415: Initial release of OpenInference instrumentation for Claude Agent SDK

  - Instruments V1 (`query()`) and V2 (`unstable_v2_prompt`, `unstable_v2_createSession`, `unstable_v2_resumeSession`) APIs
  - Produces AGENT spans for query/prompt/session-turn lifecycles
  - Produces TOOL child spans via hook injection (PreToolUse/PostToolUse/PostToolUseFailure)
  - Captures input/output values, token counts, cost, session ID, and model name
  - Supports trace configuration for masking sensitive data
  - Supports both CommonJS and ESM module loading
