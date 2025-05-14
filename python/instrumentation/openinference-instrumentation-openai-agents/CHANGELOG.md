# Changelog

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.11...python-openinference-instrumentation-openai-agents-v0.1.12) (2025-05-14)


### Features

* **openai-agents:** add support for exclusive processor configuration ([#1586](https://github.com/Arize-ai/openinference/issues/1586)) ([47c2ac3](https://github.com/Arize-ai/openinference/commit/47c2ac350a113bf7df45fbcebdfc19504e73723c))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.10...python-openinference-instrumentation-openai-agents-v0.1.11) (2025-05-11)


### Bug Fixes

* ruff formating fix & bump ruff version in dev requirements ([#1600](https://github.com/Arize-ai/openinference/issues/1600)) ([076bb79](https://github.com/Arize-ai/openinference/commit/076bb7966d44fccdb2ab94e6f379ef4ae22c39b1))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.9...python-openinference-instrumentation-openai-agents-v0.1.10) (2025-04-28)


### Bug Fixes

* update lower bound on openinference-semantic-conventions ([#1567](https://github.com/Arize-ai/openinference/issues/1567)) ([c2f428c](https://github.com/Arize-ai/openinference/commit/c2f428c5916c3dd62cf6670358f37111d4f7fd25))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.8...python-openinference-instrumentation-openai-agents-v0.1.9) (2025-04-24)


### Bug Fixes

* **openai_agents:** Set status `on_span_end`. ([#1556](https://github.com/Arize-ai/openinference/issues/1556)) ([2b53efa](https://github.com/Arize-ai/openinference/commit/2b53efa491d81ab5852387f5a4d2e87972262616))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.7...python-openinference-instrumentation-openai-agents-v0.1.8) (2025-04-11)


### Bug Fixes

* increased minimum supported version of openinference-instrumentation to 0.1.27 ([#1507](https://github.com/Arize-ai/openinference/issues/1507)) ([a55edfa](https://github.com/Arize-ai/openinference/commit/a55edfa8900c1f36a73385c7d03f91cffadd85c4))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.6...python-openinference-instrumentation-openai-agents-v0.1.7) (2025-04-02)


### Features

* cached and reasoning tokens from response api ([#1461](https://github.com/Arize-ai/openinference/issues/1461)) ([a9f257c](https://github.com/Arize-ai/openinference/commit/a9f257c1dee46eb18ed32f463bdc50cc7cab60fe))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.5...python-openinference-instrumentation-openai-agents-v0.1.6) (2025-04-02)


### Features

* capture result from MCPListToolsSpanData ([#1458](https://github.com/Arize-ai/openinference/issues/1458)) ([66abe50](https://github.com/Arize-ai/openinference/commit/66abe50f187a45ce11fb64f3399b52a3139fe115))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.4...python-openinference-instrumentation-openai-agents-v0.1.5) (2025-03-28)


### Bug Fixes

* handle error: invalid type dict in attribute 'output.value' value sequence ([#1443](https://github.com/Arize-ai/openinference/issues/1443)) ([34bacfd](https://github.com/Arize-ai/openinference/commit/34bacfd9369dfb098e931cf20982b286fcb7fbea))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.3...python-openinference-instrumentation-openai-agents-v0.1.4) (2025-03-25)


### Bug Fixes

* get attributes from GenerationSpanData (i.e. chat completions api) ([#1426](https://github.com/Arize-ai/openinference/issues/1426)) ([c0f238d](https://github.com/Arize-ai/openinference/commit/c0f238d36f18bdec0062e84ca4e53a66c63508e0))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.2...python-openinference-instrumentation-openai-agents-v0.1.3) (2025-03-24)


### Bug Fixes

* openai agent with int tool return ([#1419](https://github.com/Arize-ai/openinference/issues/1419)) ([1bb75a9](https://github.com/Arize-ai/openinference/commit/1bb75a94999bbe8615cdc7a5490fb2668833742f))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.1...python-openinference-instrumentation-openai-agents-v0.1.2) (2025-03-21)


### Bug Fixes

* add span attributes for FunctionSpanData ([#1408](https://github.com/Arize-ai/openinference/issues/1408)) ([48d1a35](https://github.com/Arize-ai/openinference/commit/48d1a3549eb8dda55e941cab867d9581a96fdf33))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-agents-v0.1.0...python-openinference-instrumentation-openai-agents-v0.1.1) (2025-03-17)


### Bug Fixes

* attach span to otel context on span start ([#1373](https://github.com/Arize-ai/openinference/issues/1373)) ([b44809f](https://github.com/Arize-ai/openinference/commit/b44809f1c460dd3a9bee4a9b068e6c275fecf9b4))
* classify handoff span as tool span ([#1374](https://github.com/Arize-ai/openinference/issues/1374)) ([e75a444](https://github.com/Arize-ai/openinference/commit/e75a444d766d900ec3bc78b9d257453fb0e586d1))

## 0.1.0 (2025-03-14)


### Features

* openai-agents instrumentation ([#1350](https://github.com/Arize-ai/openinference/issues/1350)) ([9afbad3](https://github.com/Arize-ai/openinference/commit/9afbad3100d68601a2f9265fe20985a34f80e04b))
