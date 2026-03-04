# Changelog

## [0.1.0] - 2025-02-23

### Added

- Initial release.
- Instrumentation for `claude_agent_sdk.query()`:
  - One CHAIN span per agent run with input (prompt, options) and output (message summary).
  - Entry points for `opentelemetry_instrumentor` and `openinference_instrumentor` as `claude_agent_sdk`.
