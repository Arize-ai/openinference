# Changelog

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.12...python-openinference-instrumentation-bedrock-v0.1.13) (2025-01-29)


### Features

* support bedrock anthropic messages via invoke_model_with_stream_response ([#1240](https://github.com/Arize-ai/openinference/issues/1240)) ([6047451](https://github.com/Arize-ai/openinference/commit/6047451290578402e3d9c6544067c7845c8ec134))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.11...python-openinference-instrumentation-bedrock-v0.1.12) (2024-10-31)


### Bug Fixes

* increase version lower bound for openinference-instrumentation ([#1012](https://github.com/Arize-ai/openinference/issues/1012)) ([3236d27](https://github.com/Arize-ai/openinference/commit/3236d2733a46b84d693ddb7092209800cde8cc34))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.10...python-openinference-instrumentation-bedrock-v0.1.11) (2024-08-27)


### Bug Fixes

* **bedrock:** ensure bedrock instrumentation does not break runtime for BedrockEmbeddings model from langchain ([#975](https://github.com/Arize-ai/openinference/issues/975)) ([fbb78cd](https://github.com/Arize-ai/openinference/commit/fbb78cdf13cc895add911575dc7fb400afafff7d))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.9...python-openinference-instrumentation-bedrock-v0.1.10) (2024-08-10)


### Documentation

* TraceConfig and context attributes ([#793](https://github.com/Arize-ai/openinference/issues/793)) ([d3808c4](https://github.com/Arize-ai/openinference/commit/d3808c4bea3f6a4c72d3a7ea09b54e78072be6fd))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.8...python-openinference-instrumentation-bedrock-v0.1.9) (2024-08-07)


### Features

* Capture images tracing converse API ([#753](https://github.com/Arize-ai/openinference/issues/753)) ([2a2fe15](https://github.com/Arize-ai/openinference/commit/2a2fe15f2b48fe67b14974137c105606072394f3))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.7...python-openinference-instrumentation-bedrock-v0.1.8) (2024-08-03)


### Features

* Enable config propagation ([#741](https://github.com/Arize-ai/openinference/issues/741)) ([16cdc6b](https://github.com/Arize-ai/openinference/commit/16cdc6b71fb14728a3eca7db27a55b68187cb4aa))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.6...python-openinference-instrumentation-bedrock-v0.1.7) (2024-07-18)


### Features

* **bedrock:** add converse api support for bedrock ([#585](https://github.com/Arize-ai/openinference/issues/585)) ([b41cab7](https://github.com/Arize-ai/openinference/commit/b41cab7ebc1abd730cf26f8e9d7cafce39b59054))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.5...python-openinference-instrumentation-bedrock-v0.1.6) (2024-05-21)


### Bug Fixes

* improve bedrock support for non-ascii characters ([#475](https://github.com/Arize-ai/openinference/issues/475)) ([e6d1f0a](https://github.com/Arize-ai/openinference/commit/e6d1f0acca9ea5d5f00fa10a809a771deb3ff605))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.4...python-openinference-instrumentation-bedrock-v0.1.5) (2024-05-11)


### Features

* add support for meta vendor in openinference-instrumentation-bedrock ([#448](https://github.com/Arize-ai/openinference/issues/448)) ([958c385](https://github.com/Arize-ai/openinference/commit/958c385b1d8de70698bc2e8368cef36a8e5acf8f))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.3...python-openinference-instrumentation-bedrock-v0.1.4) (2024-05-10)


### Features

* Make Bedrock instrumentation read from attributes from context ([#433](https://github.com/Arize-ai/openinference/issues/433)) ([69e8277](https://github.com/Arize-ai/openinference/commit/69e82773ad4bd6c386831c1b35310996a6967b64))


### Documentation

* Improve README ([#278](https://github.com/Arize-ai/openinference/issues/278)) ([defc847](https://github.com/Arize-ai/openinference/commit/defc847bf66dbdf6c38636ee3f88f2b0584fc035))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.2...python-openinference-instrumentation-bedrock-v0.1.3) (2024-03-13)


### Features

* add support for python 3.12 ([#271](https://github.com/Arize-ai/openinference/issues/271)) ([0556d72](https://github.com/Arize-ai/openinference/commit/0556d72997ef607545488112cde881e8660bf5db))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.1...python-openinference-instrumentation-bedrock-v0.1.2) (2024-03-12)


### Bug Fixes

* Add token counts to bedrock instrumentation ([#270](https://github.com/Arize-ai/openinference/issues/270)) ([29b0ac9](https://github.com/Arize-ai/openinference/commit/29b0ac9d643c47bb7c6fd62d4cf581dd8157291c))


### Documentation

* Convert READMEs to Markdown ([#227](https://github.com/Arize-ai/openinference/issues/227)) ([e4bcf4b](https://github.com/Arize-ai/openinference/commit/e4bcf4b86f27cc119a77f551811f9142ec6075ce))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-bedrock-v0.1.0...python-openinference-instrumentation-bedrock-v0.1.1) (2024-01-26)


### Bug Fixes

* bedrock README.rst ([#152](https://github.com/Arize-ai/openinference/issues/152)) ([50a29ef](https://github.com/Arize-ai/openinference/commit/50a29eff13afb88f0b7a6bdce4a1cc4996e385b7))

## 0.1.0 (2024-01-26)


### Features

* Add boto3 instrumentation ([#115](https://github.com/Arize-ai/openinference/issues/115)) ([01aad1f](https://github.com/Arize-ai/openinference/commit/01aad1fa63c92dbb175b68223babe433547bff48))
