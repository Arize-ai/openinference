# Changelog

## [0.1.19](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.18...python-openinference-instrumentation-anthropic-v0.1.19) (2025-08-18)


### Features

* **anthropic:** Added support for anthropic sdk input images rendering  ([#2091](https://github.com/Arize-ai/openinference/issues/2091)) ([d2684b4](https://github.com/Arize-ai/openinference/commit/d2684b4b20e3b0199051c553580a8b27b673727f))

## [0.1.18](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.17...python-openinference-instrumentation-anthropic-v0.1.18) (2025-05-19)


### Features

* **anthropic:** add stream wrapper and tests ([#1572](https://github.com/Arize-ai/openinference/issues/1572)) ([918aa01](https://github.com/Arize-ai/openinference/commit/918aa017441fd4c8cffdbcaab287913349a41a60))

## [0.1.17](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.16...python-openinference-instrumentation-anthropic-v0.1.17) (2025-04-28)


### Bug Fixes

* update lower bound on openinference-semantic-conventions ([#1567](https://github.com/Arize-ai/openinference/issues/1567)) ([c2f428c](https://github.com/Arize-ai/openinference/commit/c2f428c5916c3dd62cf6670358f37111d4f7fd25))

## [0.1.16](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.15...python-openinference-instrumentation-anthropic-v0.1.16) (2025-04-11)


### Bug Fixes

* increased minimum supported version of openinference-instrumentation to 0.1.27 ([#1507](https://github.com/Arize-ai/openinference/issues/1507)) ([a55edfa](https://github.com/Arize-ai/openinference/commit/a55edfa8900c1f36a73385c7d03f91cffadd85c4))

## [0.1.15](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.14...python-openinference-instrumentation-anthropic-v0.1.15) (2025-04-03)


### Features

* instrumentation-anthropic cache token counts ([#1465](https://github.com/Arize-ai/openinference/issues/1465)) ([d6765e0](https://github.com/Arize-ai/openinference/commit/d6765e0edd455fb879ccf0b58ea7d3dfaabeabf0))

## [0.1.14](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.13...python-openinference-instrumentation-anthropic-v0.1.14) (2025-03-25)


### Bug Fixes

* include cache tokens in prompt tokens for anthropic ([#1429](https://github.com/Arize-ai/openinference/issues/1429)) ([abd36c4](https://github.com/Arize-ai/openinference/commit/abd36c45ea4ff966b58eccee42de252bc876d5ab))

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.12...python-openinference-instrumentation-anthropic-v0.1.13) (2025-03-14)


### Documentation

* fix license to be openinference ([#1353](https://github.com/Arize-ai/openinference/issues/1353)) ([85d435b](https://github.com/Arize-ai/openinference/commit/85d435be3af3de5424494cfbdd654454688b7377))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.11...python-openinference-instrumentation-anthropic-v0.1.12) (2025-02-18)


### Features

* define openinference_instrumentor entry points for all libraries ([#1290](https://github.com/Arize-ai/openinference/issues/1290)) ([4b69fdc](https://github.com/Arize-ai/openinference/commit/4b69fdc13210048009e51639b01e7c0c9550c9d1))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.10...python-openinference-instrumentation-anthropic-v0.1.11) (2025-02-11)


### Features

* add entrypoint for use in opentelemetry-instrument ([#1278](https://github.com/Arize-ai/openinference/issues/1278)) ([2106acf](https://github.com/Arize-ai/openinference/commit/2106acfd6648804abe9b95e41a49df26a500435c))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.9...python-openinference-instrumentation-anthropic-v0.1.10) (2025-02-04)


### Bug Fixes

* support python 3.13 and drop python 3.8 ([#1263](https://github.com/Arize-ai/openinference/issues/1263)) ([5bfaa90](https://github.com/Arize-ai/openinference/commit/5bfaa90d800a8f725b3ac7444d16972ed7821738))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.8...python-openinference-instrumentation-anthropic-v0.1.9) (2024-11-21)


### Bug Fixes

* add tool id for anthropic instrumentor and serialize `content` to string if it's not a string ([#1129](https://github.com/Arize-ai/openinference/issues/1129)) ([682724c](https://github.com/Arize-ai/openinference/commit/682724ce436ef8ece5d821073e3845cc3a9d602d))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.7...python-openinference-instrumentation-anthropic-v0.1.8) (2024-11-20)


### Bug Fixes

* capture anthropic model name from response ([#1124](https://github.com/Arize-ai/openinference/issues/1124)) ([8e915f2](https://github.com/Arize-ai/openinference/commit/8e915f2589764575dea0771284c4ecf3182460ec))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.6...python-openinference-instrumentation-anthropic-v0.1.7) (2024-10-31)


### Bug Fixes

* remove anthropic dependency ([#1094](https://github.com/Arize-ai/openinference/issues/1094)) ([4dd0bba](https://github.com/Arize-ai/openinference/commit/4dd0bba7687f7ab70e0cb973ac588a850a9e99b2))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.5...python-openinference-instrumentation-anthropic-v0.1.6) (2024-10-30)


### Features

* **anthropic:** add tool json schema attributes to anthropic instrumentation ([#1087](https://github.com/Arize-ai/openinference/issues/1087)) ([907b6e5](https://github.com/Arize-ai/openinference/commit/907b6e530cb3ded377e99a7cbe7de1f35f55d39f))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.4...python-openinference-instrumentation-anthropic-v0.1.5) (2024-10-28)


### Features

* **anthropic:** add llm provider and system attributes to anthropic instrumentation ([#1084](https://github.com/Arize-ai/openinference/issues/1084)) ([32756ed](https://github.com/Arize-ai/openinference/commit/32756ed864849082f9eefc63e7d1fd8ec999d0b3))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.3...python-openinference-instrumentation-anthropic-v0.1.4) (2024-10-11)


### Features

* **anthropic:** streaming support ([#990](https://github.com/Arize-ai/openinference/issues/990)) ([f3b7b96](https://github.com/Arize-ai/openinference/commit/f3b7b96b1ddaf7194253e3233b9124c73a19840a))


### Bug Fixes

* increase version lower bound for openinference-instrumentation ([#1012](https://github.com/Arize-ai/openinference/issues/1012)) ([3236d27](https://github.com/Arize-ai/openinference/commit/3236d2733a46b84d693ddb7092209800cde8cc34))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.2...python-openinference-instrumentation-anthropic-v0.1.3) (2024-09-04)


### Bug Fixes

* **anthropic:** dynamically import anthropic ([#998](https://github.com/Arize-ai/openinference/issues/998)) ([b627f79](https://github.com/Arize-ai/openinference/commit/b627f796afd2d1499c3bcccfbc17567aa7298df8))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.1...python-openinference-instrumentation-anthropic-v0.1.2) (2024-08-28)


### Bug Fixes

* Fix input message attribute issues + toolcalling from dogfooding ([#948](https://github.com/Arize-ai/openinference/issues/948)) ([dde31f5](https://github.com/Arize-ai/openinference/commit/dde31f51755e5883561d0e9dc195cff13f38f56e))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-anthropic-v0.1.0...python-openinference-instrumentation-anthropic-v0.1.1) (2024-08-21)


### Features

* Tool calling for Anthropic instrumentor ([#939](https://github.com/Arize-ai/openinference/issues/939)) ([2566486](https://github.com/Arize-ai/openinference/commit/25664860f6226dcc4a4ef9b19e67fdc79135889b))

## 0.1.0 (2024-08-20)


### Features

* add README and release please for anthropic instrumentor ([#925](https://github.com/Arize-ai/openinference/issues/925)) ([e8b8973](https://github.com/Arize-ai/openinference/commit/e8b897357d31b2d611c80f4e2d3c5246e87fab1d))
* Anthropic instrumentation ([#878](https://github.com/Arize-ai/openinference/issues/878)) ([895eeee](https://github.com/Arize-ai/openinference/commit/895eeee6c2e7519acf5f0d6e25598d29c4f56925))
