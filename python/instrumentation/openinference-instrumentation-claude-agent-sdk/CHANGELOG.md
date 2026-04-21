# Changelog

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.0...python-openinference-instrumentation-claude-agent-sdk-v0.1.1) (2026-04-21)


### Bug Fixes

* Support Wrapt 2.x Across All Instrumentations ([#3007](https://github.com/Arize-ai/openinference/issues/3007)) ([a151b38](https://github.com/Arize-ai/openinference/commit/a151b38d36fddb559ac883e2585d6c12e58724fb))

## 0.1.0 (2026-03-04)


### Features

* **claude-agent-sdk:** Add support for Claude Agent SDK ([#2796](https://github.com/Arize-ai/openinference/issues/2796)) ([6f627e7](https://github.com/Arize-ai/openinference/commit/6f627e74a0e06e823aa593922d8d13b8d3d9aa22))

## [0.1.0] - 2025-02-23

### Added

- Initial release.
- Instrumentation for `claude_agent_sdk.query()`:
  - One CHAIN span per agent run with input (prompt, options) and output (message summary).
  - Entry points for `opentelemetry_instrumentor` and `openinference_instrumentor` as `claude_agent_sdk`.
