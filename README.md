# OpenInference

OpenInference is a set of conventions and plugins built on top of [OpenTelemetry](https://opentelemetry.io/) to enable tracing of machine AI applications. OpenInference is natively supported by [arize-phoenix](https://github.com/Arize-ai/phoenix), but can be used with any OpenTelemetry-compatible backend as well.

## Instrumentations

OpenInference provides a set of instrumentations for popular machine learning SDKs and frameworks in a variety of languages.

## JavaScript

| Package                                                                                                         | Description                                   |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [`@arizeai/openinference-semantic-conventions`](./js/packages/openinference-semantic-conventions/README.md)     | Semantic conventions for tracing of LLM Apps. |
| [`@arizeai/openinference-instrumentation-openai`](./js/packages/openinference-instrumentation-openai/README.md) | OpenInference Instrumentation for OpenAI SDK. |
