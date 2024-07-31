# Changelog

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