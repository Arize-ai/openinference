# Changelog

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
