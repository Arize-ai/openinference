# Changelog

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-ollama-v0.1.5...python-openinference-instrumentation-ollama-v0.1.6) (2026-08-27)


### Bug Fixes

* bump openinference-instrumentation minimum to &gt;=0.1.59 ([#3615](https://github.com/Arize-ai/openinference/issues/3615)) ([75168e8](https://github.com/Arize-ai/openinference/commit/75168e886ca6f9a605f3898bb566492d48c1d5dc))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-ollama-v0.1.4...python-openinference-instrumentation-ollama-v0.1.5) (2026-08-27)


### Features

* **ollama:** Add Finish Reason Attribute ([#3586](https://github.com/Arize-ai/openinference/issues/3586)) ([294241f](https://github.com/Arize-ai/openinference/commit/294241f211c7b20939e9e0daf7f7805c65435891))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-ollama-v0.1.3...python-openinference-instrumentation-ollama-v0.1.4) (2026-08-25)


### Bug Fixes

* bump openinference-semantic-conventions minimum to &gt;=0.1.33 ([#3606](https://github.com/Arize-ai/openinference/issues/3606)) ([35c7353](https://github.com/Arize-ai/openinference/commit/35c735399cc37ef395138defaa1ccb3029d71e7e))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-ollama-v0.1.2...python-openinference-instrumentation-ollama-v0.1.3) (2026-08-24)


### Documentation

* point Arize AX links at the product page with UTM parameters ([#3587](https://github.com/Arize-ai/openinference/issues/3587)) ([cae8ec9](https://github.com/Arize-ai/openinference/commit/cae8ec9615af214359d98cb552d841986a9f02e8))

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
