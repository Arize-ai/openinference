# Changelog

## [0.1.19](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.18...python-openinference-instrumentation-haystack-v0.1.19) (2025-03-14)


### Documentation

* fix license to be openinference ([#1353](https://github.com/Arize-ai/openinference/issues/1353)) ([85d435b](https://github.com/Arize-ai/openinference/commit/85d435be3af3de5424494cfbdd654454688b7377))

## [0.1.18](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.17...python-openinference-instrumentation-haystack-v0.1.18) (2025-02-22)


### Bug Fixes

* **haystack:** update haystack for compatibility with 2.10 ([#1295](https://github.com/Arize-ai/openinference/issues/1295)) ([2f6c607](https://github.com/Arize-ai/openinference/commit/2f6c6078e4e1412306bbf954e2f9ad35336f3abc))

## [0.1.17](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.16...python-openinference-instrumentation-haystack-v0.1.17) (2025-02-18)


### Features

* define openinference_instrumentor entry points for all libraries ([#1290](https://github.com/Arize-ai/openinference/issues/1290)) ([4b69fdc](https://github.com/Arize-ai/openinference/commit/4b69fdc13210048009e51639b01e7c0c9550c9d1))

## [0.1.16](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.15...python-openinference-instrumentation-haystack-v0.1.16) (2025-02-11)


### Features

* add entrypoint for use in opentelemetry-instrument ([#1278](https://github.com/Arize-ai/openinference/issues/1278)) ([2106acf](https://github.com/Arize-ai/openinference/commit/2106acfd6648804abe9b95e41a49df26a500435c))

## [0.1.15](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.14...python-openinference-instrumentation-haystack-v0.1.15) (2025-02-04)


### Bug Fixes

* support python 3.13 and drop python 3.8 ([#1263](https://github.com/Arize-ai/openinference/issues/1263)) ([5bfaa90](https://github.com/Arize-ai/openinference/commit/5bfaa90d800a8f725b3ac7444d16972ed7821738))

## [0.1.14](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.13...python-openinference-instrumentation-haystack-v0.1.14) (2025-01-17)


### Bug Fixes

* **haystack:** ensure compatibility with haystack 2.9 ([#1205](https://github.com/Arize-ai/openinference/issues/1205)) ([6ee2ebf](https://github.com/Arize-ai/openinference/commit/6ee2ebf95c88bf54b2a65dfcc04ab72d8f20a7db))

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.12...python-openinference-instrumentation-haystack-v0.1.13) (2024-10-31)


### Bug Fixes

* increase version lower bound for openinference-instrumentation ([#1012](https://github.com/Arize-ai/openinference/issues/1012)) ([3236d27](https://github.com/Arize-ai/openinference/commit/3236d2733a46b84d693ddb7092209800cde8cc34))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.11...python-openinference-instrumentation-haystack-v0.1.12) (2024-08-26)


### Bug Fixes

* **haystack:** safely get embedding model name ([#965](https://github.com/Arize-ai/openinference/issues/965)) ([3a9286b](https://github.com/Arize-ai/openinference/commit/3a9286b64d62d289d2cbaacb5672e01f4fc6fa3a))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.10...python-openinference-instrumentation-haystack-v0.1.11) (2024-08-23)


### Bug Fixes

* **haystack:** remove remaining haystack import ([#961](https://github.com/Arize-ai/openinference/issues/961)) ([fe62e3f](https://github.com/Arize-ai/openinference/commit/fe62e3f23c31f88ba99ddab2ca6b453677e7dd31))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.9...python-openinference-instrumentation-haystack-v0.1.10) (2024-08-23)


### Bug Fixes

* **haystack:** ensure haystack is not a runtime dependency ([#959](https://github.com/Arize-ai/openinference/issues/959)) ([c06813c](https://github.com/Arize-ai/openinference/commit/c06813c709331b76b5d65400eca337510d1e7ed3))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.8...python-openinference-instrumentation-haystack-v0.1.9) (2024-08-16)


### Bug Fixes

* **haystack:** ensure important attributes such as span kind are not lost ([#917](https://github.com/Arize-ai/openinference/issues/917)) ([963ba4a](https://github.com/Arize-ai/openinference/commit/963ba4acf8cc7eaad4a4f780c5e50810fb876c8a))
* **haystack:** improve heuristic for identifying component type ([#919](https://github.com/Arize-ai/openinference/issues/919)) ([bdfbbdb](https://github.com/Arize-ai/openinference/commit/bdfbbdb9c464ec0c2b730d7d70692ad346ce09f0))
* **haystack:** improve span names and mask output embeddings ([#903](https://github.com/Arize-ai/openinference/issues/903)) ([7c754f3](https://github.com/Arize-ai/openinference/commit/7c754f340982d678e9362e4da82594589e98cba7))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.7...python-openinference-instrumentation-haystack-v0.1.8) (2024-08-15)


### Features

* add re-ranker support in haystack ([#894](https://github.com/Arize-ai/openinference/issues/894)) ([88ab293](https://github.com/Arize-ai/openinference/commit/88ab29345e33508120a626374ff309d8dbd65bdb))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.6...python-openinference-instrumentation-haystack-v0.1.7) (2024-08-15)


### Bug Fixes

* haystack tool calling for `OpenAIChatGenerator` ([#876](https://github.com/Arize-ai/openinference/issues/876)) ([398e2d5](https://github.com/Arize-ai/openinference/commit/398e2d5e8cccc668a09060b252ae331d98e3c35b))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.5...python-openinference-instrumentation-haystack-v0.1.6) (2024-08-13)


### Features

* Tool calling support for HaystackInstrumentor ([#839](https://github.com/Arize-ai/openinference/issues/839)) ([96586be](https://github.com/Arize-ai/openinference/commit/96586be393a14eee0f00dc9ddd67a28a1be02d06))


### Bug Fixes

* implement prompt template attributes for prompt builder and add helper functions ([#844](https://github.com/Arize-ai/openinference/issues/844)) ([d3dcb5c](https://github.com/Arize-ai/openinference/commit/d3dcb5c3c3c2f24c06375468dc033a5e0d45779f))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.4...python-openinference-instrumentation-haystack-v0.1.5) (2024-08-10)


### Bug Fixes

* **haystack:** Adjust LLM message attributes for Haystack instrumentation to match semantic conventions ([#837](https://github.com/Arize-ai/openinference/issues/837)) ([3bde318](https://github.com/Arize-ai/openinference/commit/3bde31808bc3f05f5d1c675d7ad3fc4f15dccb7c))
* Setting of attributes crashes Phoenix and default some components to chain ([#818](https://github.com/Arize-ai/openinference/issues/818)) ([230eaef](https://github.com/Arize-ai/openinference/commit/230eaef5e46a8e72aae601745035a2c799f6799c))


### Documentation

* **haystack:** add rag example from haystack ([#812](https://github.com/Arize-ai/openinference/issues/812)) ([06e70b6](https://github.com/Arize-ai/openinference/commit/06e70b629dc5decf12a9da2f2ff197e5542344f4))
* **haystack:** web questions example ([#809](https://github.com/Arize-ai/openinference/issues/809)) ([5eaf4d8](https://github.com/Arize-ai/openinference/commit/5eaf4d8a92d7c7e9500b43d9a14d7c5f28202581))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.3...python-openinference-instrumentation-haystack-v0.1.4) (2024-08-07)


### Bug Fixes

* bump minimum version for openinference-instrumentation ([#810](https://github.com/Arize-ai/openinference/issues/810)) ([12e11ea](https://github.com/Arize-ai/openinference/commit/12e11ea405252ca35dc8d3f3a08ec5b83a08cea7))


### Documentation

* TraceConfig and context attributes ([#793](https://github.com/Arize-ai/openinference/issues/793)) ([d3808c4](https://github.com/Arize-ai/openinference/commit/d3808c4bea3f6a4c72d3a7ea09b54e78072be6fd))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.2...python-openinference-instrumentation-haystack-v0.1.3) (2024-08-06)


### Features

* **haystack:** Add support for ChatPromptBuilder and ChatOpenAIGenerator components ([#760](https://github.com/Arize-ai/openinference/issues/760)) ([375800d](https://github.com/Arize-ai/openinference/commit/375800de6e16e8bc21ce0ffdd5bb0ea98bf73999))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.1...python-openinference-instrumentation-haystack-v0.1.2) (2024-08-06)


### Features

* **haystack:** Support context attribute propagation ([#755](https://github.com/Arize-ai/openinference/issues/755)) ([1887ef9](https://github.com/Arize-ai/openinference/commit/1887ef93a45e2f8563f56923cec9e1b115f1153a))

* Add TraceConfig handling ([1887ef9](https://github.com/Arize-ai/openinference/commit/1887ef93a45e2f8563f56923cec9e1b115f1153a))


### Bug Fixes

* Use of PromptBuider without Documents ([1887ef9](https://github.com/Arize-ai/openinference/commit/1887ef93a45e2f8563f56923cec9e1b115f1153a))

* Use of Retriever without prompt query embedding ([1887ef9](https://github.com/Arize-ai/openinference/commit/1887ef93a45e2f8563f56923cec9e1b115f1153a)) 

* **haystack:** keep ruff guides consistent across instrumentors ([#705](https://github.com/Arize-ai/openinference/issues/705)) ([4293b48](https://github.com/Arize-ai/openinference/commit/4293b48f0124fafff1295e42bd5e20eb2d503a75))


## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-haystack-v0.1.0...python-openinference-instrumentation-haystack-v0.1.1) (2024-08-01)


### Bug Fixes

* **haystack:** tweak setting of attributes ([#701](https://github.com/Arize-ai/openinference/issues/701)) ([c0b31aa](https://github.com/Arize-ai/openinference/commit/c0b31aabcc92b1c7aabb4151b5f9b83d3e8b354a))

## 0.1.0 (2024-07-31)


### Features

* **haystack:** add instrumentation support ([#644](https://github.com/Arize-ai/openinference/issues/644)) ([4626113](https://github.com/Arize-ai/openinference/commit/46261138ec2fb7c80341d9f74a5916ff9b268f88))
