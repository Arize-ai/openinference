# OpenInference

OpenInference is a set of conventions and plugins that is complimentary to [OpenTelemetry](https://opentelemetry.io/) to enable tracing of machine AI applications. OpenInference is natively supported by [arize-phoenix](https://github.com/Arize-ai/phoenix), but can be used with any OpenTelemetry-compatible backend as well.

## Specification

The OpenInference specification is edited in markdown files found in the [spec directory](./spec/). It's designed to provide insight into the invocation of LLMs and the surrounding application context such as retrieval from vector stores and the usage of external tools such as search engines or APIs. The specification is transport and file-format agnostic, and is intended to be used in conjunction with other specifications such as JSON, ProtoBuf, and DataFrames.

## Instrumentation

OpenInference provides a set of instrumentations for popular machine learning SDKs and frameworks in a variety of languages.

## JavaScript

| Package                                                                                                         | Description                                   |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [`@arizeai/openinference-semantic-conventions`](./js/packages/openinference-semantic-conventions/README.md)     | Semantic conventions for tracing of LLM Apps. |
| [`@arizeai/openinference-instrumentation-openai`](./js/packages/openinference-instrumentation-openai/README.md) | OpenInference Instrumentation for OpenAI SDK. |
