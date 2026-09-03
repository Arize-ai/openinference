# Changelog

## [0.1.15](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.14...python-openinference-instrumentation-claude-agent-sdk-v0.1.15) (2026-09-03)


### Features

* **claude_agent_sdk:** Add Finish Reason Attribute ([#3657](https://github.com/Arize-ai/openinference/issues/3657)) ([5a18a20](https://github.com/Arize-ai/openinference/commit/5a18a20e733b76e0d52d7af247b42ecfe7cc23be))

## [0.1.14](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.13...python-openinference-instrumentation-claude-agent-sdk-v0.1.14) (2026-08-27)


### Bug Fixes

* bump openinference-instrumentation minimum to &gt;=0.1.59 ([#3615](https://github.com/Arize-ai/openinference/issues/3615)) ([75168e8](https://github.com/Arize-ai/openinference/commit/75168e886ca6f9a605f3898bb566492d48c1d5dc))

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.12...python-openinference-instrumentation-claude-agent-sdk-v0.1.13) (2026-08-27)


### Bug Fixes

* **claude-agent-sdk:** fold cache tokens into prompt and total counts ([#3611](https://github.com/Arize-ai/openinference/issues/3611)) ([18de978](https://github.com/Arize-ai/openinference/commit/18de978b6be9cf54c21bdf6431937b4f64b0f564))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.11...python-openinference-instrumentation-claude-agent-sdk-v0.1.12) (2026-08-25)


### Bug Fixes

* bump openinference-semantic-conventions minimum to &gt;=0.1.33 ([#3606](https://github.com/Arize-ai/openinference/issues/3606)) ([35c7353](https://github.com/Arize-ai/openinference/commit/35c735399cc37ef395138defaa1ccb3029d71e7e))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.10...python-openinference-instrumentation-claude-agent-sdk-v0.1.11) (2026-08-24)


### Documentation

* point Arize AX links at the product page with UTM parameters ([#3587](https://github.com/Arize-ai/openinference/issues/3587)) ([cae8ec9](https://github.com/Arize-ai/openinference/commit/cae8ec9615af214359d98cb552d841986a9f02e8))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.9...python-openinference-instrumentation-claude-agent-sdk-v0.1.10) (2026-08-12)


### Bug Fixes

* **claude-agent-sdk:** set llm.provider on the agent spans ([#3535](https://github.com/Arize-ai/openinference/issues/3535)) ([dd79ed9](https://github.com/Arize-ai/openinference/commit/dd79ed9fe3ba4232528ba507069d15d804f12d8c))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.8...python-openinference-instrumentation-claude-agent-sdk-v0.1.9) (2026-08-07)


### Bug Fixes

* bump openinference-semantic-conventions minimum to &gt;=0.1.31 ([#3474](https://github.com/Arize-ai/openinference/issues/3474)) ([5398a80](https://github.com/Arize-ai/openinference/commit/5398a80e9038ca53035cf61255992ca9d531b036))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-claude-agent-sdk-v0.1.7...python-openinference-instrumentation-claude-agent-sdk-v0.1.8) (2026-07-30)


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
