# Changelog

## [0.1.14](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.13...python-openinference-instrumentation-langchain-v0.1.14) (2024-03-20)


### Bug Fixes

* use start and end times from langchain ([#304](https://github.com/Arize-ai/openinference/issues/304)) ([9b92169](https://github.com/Arize-ai/openinference/commit/9b92169b57fedd99906758ca15089c1295e532c8))

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.12...python-openinference-instrumentation-langchain-v0.1.13) (2024-03-20)


### Bug Fixes

* add printer for run times ([#301](https://github.com/Arize-ai/openinference/issues/301)) ([242a50f](https://github.com/Arize-ai/openinference/commit/242a50f3f9babdb890aa3990544daea5371f3cb8))
* allow running handler in threads ([#302](https://github.com/Arize-ai/openinference/issues/302)) ([89c9464](https://github.com/Arize-ai/openinference/commit/89c946475584c31f77043139ee7812f3dd4eb748))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.11...python-openinference-instrumentation-langchain-v0.1.12) (2024-03-13)


### Features

* add support for python 3.12 ([#271](https://github.com/Arize-ai/openinference/issues/271)) ([0556d72](https://github.com/Arize-ai/openinference/commit/0556d72997ef607545488112cde881e8660bf5db))


### Bug Fixes

* metadata as json string in langchain instrumentor ([#234](https://github.com/Arize-ai/openinference/issues/234)) ([0cb0850](https://github.com/Arize-ai/openinference/commit/0cb085031350d1d88724b15fb57ec390525a7e98))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.10...python-openinference-instrumentation-langchain-v0.1.11) (2024-02-24)


### Bug Fixes

* allow instrumentation to be suppressed in langchain instrumentor ([#243](https://github.com/Arize-ai/openinference/issues/243)) ([edf20f3](https://github.com/Arize-ai/openinference/commit/edf20f390b8811f751f1b8d2a0a9814535df4a90))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.9...python-openinference-instrumentation-langchain-v0.1.10) (2024-02-24)


### Bug Fixes

* ensure langchain metadata is always json serializable ([#245](https://github.com/Arize-ai/openinference/issues/245)) ([bb6b08b](https://github.com/Arize-ai/openinference/commit/bb6b08b4789c1795921d23129a2f281c7e88f5c9))


### Documentation

* Convert READMEs to Markdown ([#227](https://github.com/Arize-ai/openinference/issues/227)) ([e4bcf4b](https://github.com/Arize-ai/openinference/commit/e4bcf4b86f27cc119a77f551811f9142ec6075ce))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.8...python-openinference-instrumentation-langchain-v0.1.9) (2024-02-15)


### Bug Fixes

* avoid langchain dependency at import time ([#222](https://github.com/Arize-ai/openinference/issues/222)) ([a3e1b52](https://github.com/Arize-ai/openinference/commit/a3e1b52a2868726d27299d7121c74fc00a26ddef))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.7...python-openinference-instrumentation-langchain-v0.1.8) (2024-02-15)


### Bug Fixes

* retain empty strings for template variables ([#213](https://github.com/Arize-ai/openinference/issues/213)) ([c262008](https://github.com/Arize-ai/openinference/commit/c2620089bc960f3778e4d0182e962c7e90a6ba5f))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.6...python-openinference-instrumentation-langchain-v0.1.7) (2024-02-15)


### Bug Fixes

* replace NaN with none in output JSON ([#211](https://github.com/Arize-ai/openinference/issues/211)) ([306241d](https://github.com/Arize-ai/openinference/commit/306241d4a08ce9d4d6c437365fdf9cb22b8a7622))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.5...python-openinference-instrumentation-langchain-v0.1.6) (2024-02-14)


### Bug Fixes

* handle BaseException ([#208](https://github.com/Arize-ai/openinference/issues/208)) ([398280b](https://github.com/Arize-ai/openinference/commit/398280b9d836f269af4e1e95e5b20c672b213f3d))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.4...python-openinference-instrumentation-langchain-v0.1.5) (2024-02-12)


### Bug Fixes

* stop attaching context for callback based instrumentors ([#192](https://github.com/Arize-ai/openinference/issues/192)) ([c05ab06](https://github.com/Arize-ai/openinference/commit/c05ab06e4529bf15953715f94bcaf4a616755d90))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.3...python-openinference-instrumentation-langchain-v0.1.4) (2024-02-08)


### Bug Fixes

* set langchain template variables as dict ([#193](https://github.com/Arize-ai/openinference/issues/193)) ([a7e2679](https://github.com/Arize-ai/openinference/commit/a7e2679f7e03a2a5813d82411d35a207383fdd31))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.2...python-openinference-instrumentation-langchain-v0.1.3) (2024-02-08)


### Bug Fixes

* **python:** parent id for async in langchain-core 0.1.21 ([#186](https://github.com/Arize-ai/openinference/issues/186)) ([9e59803](https://github.com/Arize-ai/openinference/commit/9e59803dbbb9adc8c9482a62e2d2410a0d44d01a))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.1...python-openinference-instrumentation-langchain-v0.1.2) (2024-02-08)


### Features

* **langchain:** add support for chain metadata in spans ([#175](https://github.com/Arize-ai/openinference/issues/175)) ([f218740](https://github.com/Arize-ai/openinference/commit/f2187403dccad43fe201be46ec4357ba2e1b1523))


### Bug Fixes

* JSON string attributes ([#157](https://github.com/Arize-ai/openinference/issues/157)) ([392057e](https://github.com/Arize-ai/openinference/commit/392057ecf4b601c5d8149697b4b8b3e91a2a2af6))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-langchain-v0.1.0...python-openinference-instrumentation-langchain-v0.1.1) (2024-02-01)


### Bug Fixes

* erroneous if statement ([#161](https://github.com/Arize-ai/openinference/issues/161)) ([e92afc1](https://github.com/Arize-ai/openinference/commit/e92afc16d5b0caa8fb98d6167cbe3e9263b981f0))

## 0.1.0 (2024-01-26)


### Features

* langchain instrumentor ([#138](https://github.com/Arize-ai/openinference/issues/138)) ([61094f6](https://github.com/Arize-ai/openinference/commit/61094f606dc0a6961fc566d8d45b27967a14c59c))
