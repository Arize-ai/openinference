# Changelog

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.12...python-openinference-instrumentation-litellm-v0.1.13) (2025-03-17)


### Bug Fixes

* **liteLLM:** support sync stream ([#1307](https://github.com/Arize-ai/openinference/issues/1307)) ([5c04fa0](https://github.com/Arize-ai/openinference/commit/5c04fa0e10cd95db50e11b1be9afa0f2c3a39aa5))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.11...python-openinference-instrumentation-litellm-v0.1.12) (2025-03-14)


### Documentation

* fix license to be openinference ([#1353](https://github.com/Arize-ai/openinference/issues/1353)) ([85d435b](https://github.com/Arize-ai/openinference/commit/85d435be3af3de5424494cfbdd654454688b7377))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.10...python-openinference-instrumentation-litellm-v0.1.11) (2025-02-18)


### Features

* define openinference_instrumentor entry points for all libraries ([#1290](https://github.com/Arize-ai/openinference/issues/1290)) ([4b69fdc](https://github.com/Arize-ai/openinference/commit/4b69fdc13210048009e51639b01e7c0c9550c9d1))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.9...python-openinference-instrumentation-litellm-v0.1.10) (2025-02-11)


### Features

* add entrypoint for use in opentelemetry-instrument ([#1278](https://github.com/Arize-ai/openinference/issues/1278)) ([2106acf](https://github.com/Arize-ai/openinference/commit/2106acfd6648804abe9b95e41a49df26a500435c))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.8...python-openinference-instrumentation-litellm-v0.1.9) (2025-02-05)


### Bug Fixes

* use safe_json_dumps for invocation parameters in litellm instrumentor ([#1269](https://github.com/Arize-ai/openinference/issues/1269)) ([650dbb9](https://github.com/Arize-ai/openinference/commit/650dbb9f83ce7e94329d159819033d8f86e21129))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.7...python-openinference-instrumentation-litellm-v0.1.8) (2025-02-04)


### Bug Fixes

* support python 3.13 and drop python 3.8 ([#1263](https://github.com/Arize-ai/openinference/issues/1263)) ([5bfaa90](https://github.com/Arize-ai/openinference/commit/5bfaa90d800a8f725b3ac7444d16972ed7821738))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.6...python-openinference-instrumentation-litellm-v0.1.7) (2025-02-04)


### Features

* **liteLLM:** add support for acompletion streaming (resolves [#1224](https://github.com/Arize-ai/openinference/issues/1224)) ([#1246](https://github.com/Arize-ai/openinference/issues/1246)) ([c461b98](https://github.com/Arize-ai/openinference/commit/c461b981da3ad541fcdf991cca01310cc3eab9a8))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.5...python-openinference-instrumentation-litellm-v0.1.6) (2025-01-15)


### Bug Fixes

* Add support for output messages for sync/async ([#1188](https://github.com/Arize-ai/openinference/issues/1188)) ([0bb96b6](https://github.com/Arize-ai/openinference/commit/0bb96b65ebd261445fb63ccc06da49f365dc1fa3))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.4...python-openinference-instrumentation-litellm-v0.1.5) (2024-10-31)


### Bug Fixes

* increase version lower bound for openinference-instrumentation ([#1012](https://github.com/Arize-ai/openinference/issues/1012)) ([3236d27](https://github.com/Arize-ai/openinference/commit/3236d2733a46b84d693ddb7092209800cde8cc34))


### Documentation

* litellm examples ([#681](https://github.com/Arize-ai/openinference/issues/681)) ([b6cfe69](https://github.com/Arize-ai/openinference/commit/b6cfe6933d840b2344b5c132a9d471d239af1c9d))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.3...python-openinference-instrumentation-litellm-v0.1.4) (2024-08-27)


### Features

* **liteLLM:** Implemented image support and corresponding tests ([#900](https://github.com/Arize-ai/openinference/issues/900)) ([f6d11eb](https://github.com/Arize-ai/openinference/commit/f6d11eb602f37770fbdf7ab144c03980c7f90fb7))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.2...python-openinference-instrumentation-litellm-v0.1.3) (2024-08-13)


### Features

* **liteLLM:** Added suppress tracing to litellm instrumentation ([#847](https://github.com/Arize-ai/openinference/issues/847)) ([bda858a](https://github.com/Arize-ai/openinference/commit/bda858ad332a8f9539f9a9edb77d9ede22a08960))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.1...python-openinference-instrumentation-litellm-v0.1.2) (2024-08-10)


### Features

* **liteLLM:** LiteLLM trace config and context attributes propagation ([#779](https://github.com/Arize-ai/openinference/issues/779)) ([d104695](https://github.com/Arize-ai/openinference/commit/d104695cdcebea740f98b2e26a2a5bab1a09a55f))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-litellm-v0.1.0...python-openinference-instrumentation-litellm-v0.1.1) (2024-08-07)


### Bug Fixes

* bump minimum version for openinference-instrumentation ([#810](https://github.com/Arize-ai/openinference/issues/810)) ([12e11ea](https://github.com/Arize-ai/openinference/commit/12e11ea405252ca35dc8d3f3a08ec5b83a08cea7))


### Documentation

* TraceConfig and context attributes ([#793](https://github.com/Arize-ai/openinference/issues/793)) ([d3808c4](https://github.com/Arize-ai/openinference/commit/d3808c4bea3f6a4c72d3a7ea09b54e78072be6fd))

## 0.1.0 (2024-07-31)


### Features

* **liteLLM:** instrumentation ([#641](https://github.com/Arize-ai/openinference/issues/641)) ([9870286](https://github.com/Arize-ai/openinference/commit/9870286e8ea757ca3afa2568bd286231fbaee577))

## 0.1.0 (2024-07-29)

### Features
* liteLLM functions that have been instrumented:
    - completion()
    - acompletion()
    - completion_with_retries()
    - embedding()
    - aembedding()
    - image_generation()
    - aimage_generation()
* liteLLM functions that currently don't work:
    - acompletion_with_retries() **

** Currently acompletion_with_retries() is buggy on liteLLM's part. A bug report (https://github.com/BerriAI/litellm/issues/4908) has been filed with liteLLM but for now, the instrumentation and test for acompletion_with_retries() have been commented out
