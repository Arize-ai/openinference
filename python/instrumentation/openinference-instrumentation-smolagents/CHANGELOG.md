# Changelog

## [0.1.16](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.15...python-openinference-instrumentation-smolagents-v0.1.16) (2025-08-22)


### Bug Fixes

* **smolagents:** prevent duplicate llm spans ([#2118](https://github.com/Arize-ai/openinference/issues/2118)) ([f1a2946](https://github.com/Arize-ai/openinference/commit/f1a29460344d37e7e431d2bd76bc6f724bcbb931))

## [0.1.15](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.14...python-openinference-instrumentation-smolagents-v0.1.15) (2025-08-19)


### Bug Fixes

* **smolagents:** use new token_usage structure ([#2087](https://github.com/Arize-ai/openinference/issues/2087)) ([c691712](https://github.com/Arize-ai/openinference/commit/c6917124a236eb6deff1e1faf214075662ebfc58))

## [0.1.14](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.13...python-openinference-instrumentation-smolagents-v0.1.14) (2025-07-10)


### Features

* **smolagents:** Improve Display Of Handled Errors ([#1845](https://github.com/Arize-ai/openinference/issues/1845)) ([fb7eb1f](https://github.com/Arize-ai/openinference/commit/fb7eb1fa36adb7bb5ff001dce196c32ceb244a18))

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.12...python-openinference-instrumentation-smolagents-v0.1.13) (2025-06-09)


### Bug Fixes

* **smolagents:** instrument both __call__ and generate methods for complete model tracing ([#1744](https://github.com/Arize-ai/openinference/issues/1744)) ([a963b96](https://github.com/Arize-ai/openinference/commit/a963b9619776abe79fb6719eeb9eda01850aeff5))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.11...python-openinference-instrumentation-smolagents-v0.1.12) (2025-05-28)


### Bug Fixes

* **smolagents:** include reasoning content ([#1697](https://github.com/Arize-ai/openinference/issues/1697)) ([0c8ea99](https://github.com/Arize-ai/openinference/commit/0c8ea99312874f605e1ab751e38dd13c8b0d4ea0))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.10...python-openinference-instrumentation-smolagents-v0.1.11) (2025-04-28)


### Bug Fixes

* update lower bound on openinference-semantic-conventions ([#1567](https://github.com/Arize-ai/openinference/issues/1567)) ([c2f428c](https://github.com/Arize-ai/openinference/commit/c2f428c5916c3dd62cf6670358f37111d4f7fd25))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.9...python-openinference-instrumentation-smolagents-v0.1.10) (2025-04-11)


### Bug Fixes

* increased minimum supported version of openinference-instrumentation to 0.1.27 ([#1507](https://github.com/Arize-ai/openinference/issues/1507)) ([a55edfa](https://github.com/Arize-ai/openinference/commit/a55edfa8900c1f36a73385c7d03f91cffadd85c4))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.8...python-openinference-instrumentation-smolagents-v0.1.9) (2025-04-05)


### Bug Fixes

* allow prerelease versions of smolagents ([#1416](https://github.com/Arize-ai/openinference/issues/1416)) ([8cd680f](https://github.com/Arize-ai/openinference/commit/8cd680fcb4b7d88a1223f2e07bf1edb038021fac))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.7...python-openinference-instrumentation-smolagents-v0.1.8) (2025-03-21)


### Bug Fixes

* only import exported smolagents models ([#1403](https://github.com/Arize-ai/openinference/issues/1403)) ([e175778](https://github.com/Arize-ai/openinference/commit/e175778252b0cd50d1d1fa20b53547fbf83f74cd))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.6...python-openinference-instrumentation-smolagents-v0.1.7) (2025-03-14)


### Documentation

* fix license to be openinference ([#1353](https://github.com/Arize-ai/openinference/issues/1353)) ([85d435b](https://github.com/Arize-ai/openinference/commit/85d435be3af3de5424494cfbdd654454688b7377))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.5...python-openinference-instrumentation-smolagents-v0.1.6) (2025-02-18)


### Features

* define openinference_instrumentor entry points for all libraries ([#1290](https://github.com/Arize-ai/openinference/issues/1290)) ([4b69fdc](https://github.com/Arize-ai/openinference/commit/4b69fdc13210048009e51639b01e7c0c9550c9d1))
* **smolagents:** support smolagents on python 3.13 ([#1294](https://github.com/Arize-ai/openinference/issues/1294)) ([415f57e](https://github.com/Arize-ai/openinference/commit/415f57e9cdcaf8ad4da8f73043f0fe8e64a7a1e0))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.4...python-openinference-instrumentation-smolagents-v0.1.5) (2025-02-11)


### Features

* **smolagents:** updates to latest and makes examples use opentelemetry-instrument ([#1277](https://github.com/Arize-ai/openinference/issues/1277)) ([b151bc9](https://github.com/Arize-ai/openinference/commit/b151bc9a3f8243c846c2981ade94e3d2823602e7))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.3...python-openinference-instrumentation-smolagents-v0.1.4) (2025-02-07)


### Features

* **smolagents:** add entrypoint for use in opentelemetry-instrument ([#1276](https://github.com/Arize-ai/openinference/issues/1276)) ([7d2af53](https://github.com/Arize-ai/openinference/commit/7d2af53fea2d3b7e03b20cbf056994fddc23d888))


### Bug Fixes

* **smolagents:** support new managed agents ([#1274](https://github.com/Arize-ai/openinference/issues/1274)) ([306e1c5](https://github.com/Arize-ai/openinference/commit/306e1c5caf3827433c3a2151b93f7534533bbe94))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.2...python-openinference-instrumentation-smolagents-v0.1.3) (2025-02-04)


### Bug Fixes

* **smolagents:** support smolagents nested llm input message contents ([#1238](https://github.com/Arize-ai/openinference/issues/1238)) ([51be5e4](https://github.com/Arize-ai/openinference/commit/51be5e47f4d5ae4ccf43d33a09c3475b56edf784))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.1...python-openinference-instrumentation-smolagents-v0.1.2) (2025-01-29)


### Bug Fixes

* **smolagents:** ensure tool attributes are captured successfully on new versions of smolagents ([#1243](https://github.com/Arize-ai/openinference/issues/1243)) ([03f3ceb](https://github.com/Arize-ai/openinference/commit/03f3ceb25a5adfbfc3e1f329782a11ae59fd5b42))
* **smolagents:** remove redundant smolagents llm output.value causing warning ([#1239](https://github.com/Arize-ai/openinference/issues/1239)) ([23324a4](https://github.com/Arize-ai/openinference/commit/23324a445f7e13c42b0d17bc46c4e7fdd0ed1f55))
* **smolagents:** use correct smolagents tool schema extraction function ([#1236](https://github.com/Arize-ai/openinference/issues/1236)) ([7d764fe](https://github.com/Arize-ai/openinference/commit/7d764fe1aabf1223a177eb60cfde13dec7653417))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-smolagents-v0.1.0...python-openinference-instrumentation-smolagents-v0.1.1) (2025-01-23)


### Bug Fixes

* **smolagents:** update internal smolagents import for compatibility with versions post 1.5.0 ([#1229](https://github.com/Arize-ai/openinference/issues/1229)) ([b338113](https://github.com/Arize-ai/openinference/commit/b338113b74433462db6c91d6f96fc8d5b983948d))

## 0.1.0 (2025-01-15)


### Features

* working instrumentation with smolagents ([#1184](https://github.com/Arize-ai/openinference/issues/1184)) ([a9b70ed](https://github.com/Arize-ai/openinference/commit/a9b70ed91c21535792202d6a0df4120f6095776d))

## 0.1.0 (2025-01-10)

### Features

* **smolagents:** instrumentation ([#1182](https://github.com/Arize-ai/openinference/issues/1182)) ([#1184](https://github.com/Arize-ai/openinference/pull/1184))
