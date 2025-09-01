# Changelog

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.10...python-openinference-instrumentation-crewai-v0.1.11) (2025-07-16)


### Features

* **crewai:** capture graph.node.id and graph.node.parent_id semantics ([#1794](https://github.com/Arize-ai/openinference/issues/1794)) ([4645932](https://github.com/Arize-ai/openinference/commit/4645932b68f7ed5ab3ecd8818ddad9e1011c027e))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.9...python-openinference-instrumentation-crewai-v0.1.10) (2025-05-27)


### Bug Fixes

* **crewai:** crewai default empty tasks ([#1682](https://github.com/Arize-ai/openinference/issues/1682)) ([4a47bfc](https://github.com/Arize-ai/openinference/commit/4a47bfc065b88b55bfcb7605abf66ef12a286ec9))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.8...python-openinference-instrumentation-crewai-v0.1.9) (2025-04-28)


### Bug Fixes

* update lower bound on openinference-semantic-conventions ([#1567](https://github.com/Arize-ai/openinference/issues/1567)) ([c2f428c](https://github.com/Arize-ai/openinference/commit/c2f428c5916c3dd62cf6670358f37111d4f7fd25))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.7...python-openinference-instrumentation-crewai-v0.1.8) (2025-04-11)


### Bug Fixes

* increased minimum supported version of openinference-instrumentation to 0.1.27 ([#1507](https://github.com/Arize-ai/openinference/issues/1507)) ([a55edfa](https://github.com/Arize-ai/openinference/commit/a55edfa8900c1f36a73385c7d03f91cffadd85c4))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.6...python-openinference-instrumentation-crewai-v0.1.7) (2025-03-14)


### Documentation

* fix license to be openinference ([#1353](https://github.com/Arize-ai/openinference/issues/1353)) ([85d435b](https://github.com/Arize-ai/openinference/commit/85d435be3af3de5424494cfbdd654454688b7377))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.5...python-openinference-instrumentation-crewai-v0.1.6) (2025-02-18)


### Features

* define openinference_instrumentor entry points for all libraries ([#1290](https://github.com/Arize-ai/openinference/issues/1290)) ([4b69fdc](https://github.com/Arize-ai/openinference/commit/4b69fdc13210048009e51639b01e7c0c9550c9d1))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.4...python-openinference-instrumentation-crewai-v0.1.5) (2025-02-11)


### Features

* add entrypoint for use in opentelemetry-instrument ([#1278](https://github.com/Arize-ai/openinference/issues/1278)) ([2106acf](https://github.com/Arize-ai/openinference/commit/2106acfd6648804abe9b95e41a49df26a500435c))


### Bug Fixes

* fix test failures with crew-latest ([#1282](https://github.com/Arize-ai/openinference/issues/1282)) ([e2e3dd1](https://github.com/Arize-ai/openinference/commit/e2e3dd13bf78a3ad4b0d44fc2ae2151127583dce))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.3...python-openinference-instrumentation-crewai-v0.1.4) (2025-01-17)


### Bug Fixes

* remove token on crewai kickoff chain span ([#1213](https://github.com/Arize-ai/openinference/issues/1213)) ([f015bca](https://github.com/Arize-ai/openinference/commit/f015bca24ce5757e8c7c604487c81889e3e84027))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.2...python-openinference-instrumentation-crewai-v0.1.3) (2024-10-31)


### Bug Fixes

* increase version lower bound for openinference-instrumentation ([#1012](https://github.com/Arize-ai/openinference/issues/1012)) ([3236d27](https://github.com/Arize-ai/openinference/commit/3236d2733a46b84d693ddb7092209800cde8cc34))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.1...python-openinference-instrumentation-crewai-v0.1.2) (2024-08-20)


### Features

* **crewai:** Add SpanKind and Token Count attributes ([#867](https://github.com/Arize-ai/openinference/issues/867)) ([a61e12a](https://github.com/Arize-ai/openinference/commit/a61e12a43773b933afcce28613db70fcceba43fd))
* **crewAI:** Added trace config, context attributes, suppress tracing for CrewAI ([#851](https://github.com/Arize-ai/openinference/issues/851)) ([4ad22fa](https://github.com/Arize-ai/openinference/commit/4ad22fac38e051ea12dd53936f40741717743171))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-crewai-v0.1.0...python-openinference-instrumentation-crewai-v0.1.1) (2024-08-07)


### Bug Fixes

* bump minimum version for openinference-instrumentation ([#810](https://github.com/Arize-ai/openinference/issues/810)) ([12e11ea](https://github.com/Arize-ai/openinference/commit/12e11ea405252ca35dc8d3f3a08ec5b83a08cea7))


### Documentation

* TraceConfig and context attributes ([#793](https://github.com/Arize-ai/openinference/issues/793)) ([d3808c4](https://github.com/Arize-ai/openinference/commit/d3808c4bea3f6a4c72d3a7ea09b54e78072be6fd))

## 0.1.0 (2024-07-31)


### Features

* **crewAI:** instrumentation ([#622](https://github.com/Arize-ai/openinference/issues/622)) ([7ddbe11](https://github.com/Arize-ai/openinference/commit/7ddbe1100efb53bc7a3812b658e8cfd31b6cefcd))
