# Changelog

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.7...python-openinference-instrumentation-claude-agent-sdk-v0.1.8) (2026-07-24)


### Documentation

* link Arize AX alongside Phoenix across all READMEs ([#3330](https://github.com/Arize-ai/openinference/issues/3330)) ([0433526](https://github.com/Arize-ai/openinference/commit/0433526b048474195b4f354e5df6bfea2db4804d))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.6...python-openinference-instrumentation-claude-agent-sdk-v0.1.7) (2026-06-24)


### Bug Fixes

* **claude_agent_sdk:** Capture Thinking Blocks as Reasoning Message Content ([#3201](https://github.com/Arize-ai/openinference/issues/3201)) ([5b49859](https://github.com/Arize-ai/openinference/commit/5b49859f020189bf0835abde44f751c2a732ba8a))
* **claude_agent_sdk:** Invalid Parent for Subagent Tool in PreToolUse Hook ([#3266](https://github.com/Arize-ai/openinference/issues/3266)) ([86ac843](https://github.com/Arize-ai/openinference/commit/86ac84328f457c4aa2e4bf5356460dc03ba14003))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.5...python-openinference-instrumentation-claude-agent-sdk-v0.1.6) (2026-06-11)


### Bug Fixes

* **claude_agent_sdk:** Preserve Propagated Session ID ([#3233](https://github.com/Arize-ai/openinference/issues/3233)) ([35738c0](https://github.com/Arize-ai/openinference/commit/35738c0a13323298b62e6bbc0192a34157135171))
* **claude_agent_sdk:** Record Real Tool Error Content on Failed Tool Spans ([#3139](https://github.com/Arize-ai/openinference/issues/3139)) ([06d8eed](https://github.com/Arize-ai/openinference/commit/06d8eedae5e8c7425547c1be5d7f62c72cdb14b6))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.4...python-openinference-instrumentation-claude-agent-sdk-v0.1.5) (2026-05-29)


### Bug Fixes

* **claude_agent_sdk:** pick max-output-tokens model from modelUsage ([#3140](https://github.com/Arize-ai/openinference/issues/3140)) ([5ed6344](https://github.com/Arize-ai/openinference/commit/5ed6344a516bc434338a19b385576ebce0cdf130))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.3...python-openinference-instrumentation-claude-agent-sdk-v0.1.4) (2026-05-18)


### Bug Fixes

* bump openinference-instrumentation minimum to &gt;=0.1.51 ([#3110](https://github.com/Arize-ai/openinference/issues/3110)) ([bae43ff](https://github.com/Arize-ai/openinference/commit/bae43ff5676fbc4d3a666a15fb3bc50fb73316da))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.2...python-openinference-instrumentation-claude-agent-sdk-v0.1.3) (2026-05-14)


### Bug Fixes

* bump openinference-instrumentation minimum to &gt;=0.1.50 ([#3084](https://github.com/Arize-ai/openinference/issues/3084)) ([8a96ad7](https://github.com/Arize-ai/openinference/commit/8a96ad776e723dc1de497b28b25fbdc5e0b12355))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.1...python-openinference-instrumentation-claude-agent-sdk-v0.1.2) (2026-05-10)


### Bug Fixes

* bump openinference-instrumentation minimum to &gt;=0.1.48 ([#3042](https://github.com/Arize-ai/openinference/issues/3042)) ([298e3bf](https://github.com/Arize-ai/openinference/commit/298e3bf2b75717bd5c7238a83ac86ba3fe419297))
* bump openinference-instrumentation minimum to &gt;=0.1.49 ([#3063](https://github.com/Arize-ai/openinference/issues/3063)) ([6fbe906](https://github.com/Arize-ai/openinference/commit/6fbe9061d919251420d4c96333c12027f6348fcf))

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
