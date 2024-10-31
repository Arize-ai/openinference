# Changelog

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
