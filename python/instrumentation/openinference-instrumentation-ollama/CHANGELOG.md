# Changelog

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-ollama-v0.1.1...python-openinference-instrumentation-ollama-v0.1.2) (2026-08-07)


### Bug Fixes

* document PyPI links for new instrumentors ([#3524](https://github.com/Arize-ai/openinference/issues/3524)) ([7abfb85](https://github.com/Arize-ai/openinference/commit/7abfb850f2a93de4527cccf805494579e3348cee))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-ollama-v0.1.0...python-openinference-instrumentation-ollama-v0.1.1) (2026-08-07)


### Features

* **ollama:** add Ollama instrumentor ([#3348](https://github.com/Arize-ai/openinference/issues/3348)) ([237ce2b](https://github.com/Arize-ai/openinference/commit/237ce2b413e89782ad93431d39581a1ee44cad95))

## [0.1.0] - 2026-08-06

### Added

- Initial release.
- Instrumentation for `ollama.chat`, `ollama.Client.chat`, and `ollama.AsyncClient.chat`, including streaming and tool calls.
- Entry points for `opentelemetry_instrumentor` and `openinference_instrumentor` as `ollama`.
