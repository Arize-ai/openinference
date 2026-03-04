# Changelog

## 0.1.0 (2026-03-04)


### Features

* **claude-agent-sdk:** Add support for Claude Agent SDK ([#2796](https://github.com/Arize-ai/openinference/issues/2796)) ([6f627e7](https://github.com/Arize-ai/openinference/commit/6f627e74a0e06e823aa593922d8d13b8d3d9aa22))

## [0.1.0] - 2025-02-23

### Added

- Initial release.
- Instrumentation for `claude_agent_sdk.query()`:
  - One CHAIN span per agent run with input (prompt, options) and output (message summary).
  - Entry points for `opentelemetry_instrumentor` and `openinference_instrumentor` as `claude_agent_sdk`.
