# Changelog

## [0.1.14](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.13...python-openinference-instrumentation-openai-v0.1.14) (2024-08-16)


### Features

* Added tools attribute ([#904](https://github.com/Arize-ai/openinference/issues/904)) ([f1eb980](https://github.com/Arize-ai/openinference/commit/f1eb980a4a91d832c80252b254bf94a273c79031))

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.12...python-openinference-instrumentation-openai-v0.1.13) (2024-08-10)


### Documentation

* TraceConfig and context attributes ([#793](https://github.com/Arize-ai/openinference/issues/793)) ([d3808c4](https://github.com/Arize-ai/openinference/commit/d3808c4bea3f6a4c72d3a7ea09b54e78072be6fd))
* update openai readme ([#751](https://github.com/Arize-ai/openinference/issues/751)) ([10f20c2](https://github.com/Arize-ai/openinference/commit/10f20c2dd68f0ee8c90d8e43c0d71b408230dd66))
* update openai readme ([#754](https://github.com/Arize-ai/openinference/issues/754)) ([4b8b967](https://github.com/Arize-ai/openinference/commit/4b8b96799b255d46d95a201dd7188f545c9d1228))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.11...python-openinference-instrumentation-openai-v0.1.12) (2024-08-02)


### Bug Fixes

* set higher lower-bound for OpenInference dependency ([#739](https://github.com/Arize-ai/openinference/issues/739)) ([08f9bef](https://github.com/Arize-ai/openinference/commit/08f9bef9391856b5d4f4dbb69a2c2867fd47bc51))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.10...python-openinference-instrumentation-openai-v0.1.11) (2024-08-01)


### Bug Fixes

* Rename base tracer and masked span ([#693](https://github.com/Arize-ai/openinference/issues/693)) ([861ea4b](https://github.com/Arize-ai/openinference/commit/861ea4ba45cf02a1d0519a7cd2c5c6ca5d74115b))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.9...python-openinference-instrumentation-openai-v0.1.10) (2024-07-31)


### Features

* Move attribute censorship based on config to common instrumentation ([#679](https://github.com/Arize-ai/openinference/issues/679)) ([04f885a](https://github.com/Arize-ai/openinference/commit/04f885a5934af6fe885e7498332785da110cb500))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.8...python-openinference-instrumentation-openai-v0.1.9) (2024-07-30)


### Features

* Add TraceConfig handling  ([#640](https://github.com/Arize-ai/openinference/issues/640)) ([008f956](https://github.com/Arize-ai/openinference/commit/008f956d867fc3effaa0b75019159a15f4709322))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.7...python-openinference-instrumentation-openai-v0.1.8) (2024-07-17)


### Bug Fixes

* **openai:** missing span when stream response is used as context manager ([#591](https://github.com/Arize-ai/openinference/issues/591)) ([ee1fd0e](https://github.com/Arize-ai/openinference/commit/ee1fd0ecfc4616c481a3b81e3e2eebf5858e5d8a))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.6...python-openinference-instrumentation-openai-v0.1.7) (2024-06-27)


### Features

* Add multimodal content arrays to llm messages ([#542](https://github.com/Arize-ai/openinference/issues/542)) ([916040d](https://github.com/Arize-ai/openinference/commit/916040d50bd867c8d9fe34638b2e8b2dfca4d22c))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.5...python-openinference-instrumentation-openai-v0.1.6) (2024-05-21)


### Bug Fixes

* improve openai support for non-ascii characters ([#480](https://github.com/Arize-ai/openinference/issues/480)) ([5b9dd64](https://github.com/Arize-ai/openinference/commit/5b9dd64e4063e5d7ecf357fdba0faa70acaa1a25))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.4...python-openinference-instrumentation-openai-v0.1.5) (2024-05-09)


### Features

* add tool calling support for sync and async chat completions for mistralai instrumentation ([#313](https://github.com/Arize-ai/openinference/issues/313)) ([9889164](https://github.com/Arize-ai/openinference/commit/9889164b4dd815cdb044d6f40a9506a02adf38c2))
* OpenAI instrumentation to capture context attributes ([#415](https://github.com/Arize-ai/openinference/issues/415)) ([8e0cab9](https://github.com/Arize-ai/openinference/commit/8e0cab90c10a4e74270eacca7e0cc9271543fe2a))


### Bug Fixes

* Bump openinference-instrumentation requirement to avoid yanked release ([#426](https://github.com/Arize-ai/openinference/issues/426)) ([71b28f2](https://github.com/Arize-ai/openinference/commit/71b28f29c1331be89e0e82278b16fee2f17a0a9e))


### Documentation

* **openai:** flesh out README ([#290](https://github.com/Arize-ai/openinference/issues/290)) ([70adfae](https://github.com/Arize-ai/openinference/commit/70adfae642e0e0c1f207d237766ef54bec587236))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.3...python-openinference-instrumentation-openai-v0.1.4) (2024-03-13)


### Features

* add support for python 3.12 ([#271](https://github.com/Arize-ai/openinference/issues/271)) ([0556d72](https://github.com/Arize-ai/openinference/commit/0556d72997ef607545488112cde881e8660bf5db))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.2...python-openinference-instrumentation-openai-v0.1.3) (2024-02-24)


### Features

* add status description for openai instrumentor ([#235](https://github.com/Arize-ai/openinference/issues/235)) ([4ff392e](https://github.com/Arize-ai/openinference/commit/4ff392e2a67883b9cdcf87b54e2de4d99e2007b0))


### Bug Fixes

* missing dependency `typing-extensions` ([#220](https://github.com/Arize-ai/openinference/issues/220)) ([5b8e2e0](https://github.com/Arize-ai/openinference/commit/5b8e2e0bc2b5abe48ffbcf3c9ccc38e2e6d33d76))


### Documentation

* Convert READMEs to Markdown ([#227](https://github.com/Arize-ai/openinference/issues/227)) ([e4bcf4b](https://github.com/Arize-ai/openinference/commit/e4bcf4b86f27cc119a77f551811f9142ec6075ce))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.1...python-openinference-instrumentation-openai-v0.1.2) (2024-02-15)


### Bug Fixes

* avoid openai dependency at import time ([#216](https://github.com/Arize-ai/openinference/issues/216)) ([3bfc23c](https://github.com/Arize-ai/openinference/commit/3bfc23c2c41a06262096ae02df04d96fec471e6c))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-openai-v0.1.0...python-openinference-instrumentation-openai-v0.1.1) (2024-02-08)


### Bug Fixes

* **python:** streaming with_raw_response in openai 1.8.0 ([#181](https://github.com/Arize-ai/openinference/issues/181)) ([6c8fcf0](https://github.com/Arize-ai/openinference/commit/6c8fcf0cf10aaacb3db777e906741ea2ea3496ac))

## 0.1.0 (2024-01-11)


### Features

* **python:** openai instrumentator ([#35](https://github.com/Arize-ai/openinference/issues/35)) ([764f781](https://github.com/Arize-ai/openinference/commit/764f781081b8447412e872445716e115f4ef38aa))
