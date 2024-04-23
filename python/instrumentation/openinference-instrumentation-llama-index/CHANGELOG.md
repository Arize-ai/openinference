# Changelog

## [1.2.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v1.2.1...python-openinference-instrumentation-llama-index-v1.2.2) (2024-04-22)


### Bug Fixes

* convert numpy values for protobuf ([#394](https://github.com/Arize-ai/openinference/issues/394)) ([eccda65](https://github.com/Arize-ai/openinference/commit/eccda6510e2bbdfb6ecb7779bb7adcab77e070d3))

## [1.2.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v1.2.0...python-openinference-instrumentation-llama-index-v1.2.1) (2024-03-18)


### Bug Fixes

* for streaming use `response_txt` as output_value for llama-index ([#292](https://github.com/Arize-ai/openinference/issues/292)) ([1906ef2](https://github.com/Arize-ai/openinference/commit/1906ef2cf29b02bec5c76ba07021917f8dedc2f1))


### Documentation

* quick start demo for llama-index ([#285](https://github.com/Arize-ai/openinference/issues/285)) ([60f0671](https://github.com/Arize-ai/openinference/commit/60f06710b62828852ac5cc686e05567b75fd38a0))

## [1.2.0](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v1.1.1...python-openinference-instrumentation-llama-index-v1.2.0) (2024-03-13)


### Features

* add support for python 3.12 ([#271](https://github.com/Arize-ai/openinference/issues/271)) ([0556d72](https://github.com/Arize-ai/openinference/commit/0556d72997ef607545488112cde881e8660bf5db))

## [1.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v1.1.0...python-openinference-instrumentation-llama-index-v1.1.1) (2024-03-04)


### Bug Fixes

* concurrent tracing for llama-index ([#249](https://github.com/Arize-ai/openinference/issues/249)) ([46c9d5a](https://github.com/Arize-ai/openinference/commit/46c9d5a0cd8cc7aecde671fc1efb0a9b1fde51e9))

## [1.1.0](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v1.0.1...python-openinference-instrumentation-llama-index-v1.1.0) (2024-02-24)


### Features

* add status description to llama-index instrumentor ([#237](https://github.com/Arize-ai/openinference/issues/237)) ([d11f815](https://github.com/Arize-ai/openinference/commit/d11f815628a34dbf0015c07d9fd9321eebeca937))


### Bug Fixes

* allow instrumentation to be suppressed in llama-index instrumentor ([#241](https://github.com/Arize-ai/openinference/issues/241)) ([891a83a](https://github.com/Arize-ai/openinference/commit/891a83a21439ff698922d6332b749ec1b0f25e8b))


### Documentation

* Convert READMEs to Markdown ([#227](https://github.com/Arize-ai/openinference/issues/227)) ([e4bcf4b](https://github.com/Arize-ai/openinference/commit/e4bcf4b86f27cc119a77f551811f9142ec6075ce))

## [1.0.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v1.0.0...python-openinference-instrumentation-llama-index-v1.0.1) (2024-02-15)


### Bug Fixes

* avoid llama-index dependency at import time ([#224](https://github.com/Arize-ai/openinference/issues/224)) ([09db813](https://github.com/Arize-ai/openinference/commit/09db8132e055daa2c57765a27fb8b18939aa726e))
* missing dependency `typing-extensions` ([#217](https://github.com/Arize-ai/openinference/issues/217)) ([bd6f049](https://github.com/Arize-ai/openinference/commit/bd6f0496b3999afc43b49d74f59c51ba71032001))

## [1.0.0](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v0.1.3...python-openinference-instrumentation-llama-index-v1.0.0) (2024-02-13)


### ⚠ BREAKING CHANGES

* **llama_index:** support 0.10.0 ([#201](https://github.com/Arize-ai/openinference/issues/201))

### Features

* **llama_index:** support 0.10.0 ([#201](https://github.com/Arize-ai/openinference/issues/201)) ([b22b435](https://github.com/Arize-ai/openinference/commit/b22b435abbc3075353b6d6883ff6c973ad79feca))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v0.1.2...python-openinference-instrumentation-llama-index-v0.1.3) (2024-02-12)


### Bug Fixes

* stop attaching context for callback based instrumentors ([#192](https://github.com/Arize-ai/openinference/issues/192)) ([c05ab06](https://github.com/Arize-ai/openinference/commit/c05ab06e4529bf15953715f94bcaf4a616755d90))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v0.1.1...python-openinference-instrumentation-llama-index-v0.1.2) (2024-02-09)


### Bug Fixes

* JSON string attributes ([#157](https://github.com/Arize-ai/openinference/issues/157)) ([392057e](https://github.com/Arize-ai/openinference/commit/392057ecf4b601c5d8149697b4b8b3e91a2a2af6))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-llama-index-v0.1.0...python-openinference-instrumentation-llama-index-v0.1.1) (2024-01-27)


### Bug Fixes

* enforce explicit parent context at span creation for llama-index ([#155](https://github.com/Arize-ai/openinference/issues/155)) ([266ed67](https://github.com/Arize-ai/openinference/commit/266ed679ae44d5de12c85eda18cc9842355b9d46))

## 0.1.0 (2024-01-22)


### Features

* llama-index instrumentor using callback handler ([#121](https://github.com/Arize-ai/openinference/issues/121)) ([b0734c1](https://github.com/Arize-ai/openinference/commit/b0734c181e5a5c0e06d3a76bcbca893cd8dece0d))
