# Changelog

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
