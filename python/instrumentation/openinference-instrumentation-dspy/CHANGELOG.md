# Changelog

## [0.1.16](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.15...python-openinference-instrumentation-dspy-v0.1.16) (2025-02-04)


### Bug Fixes

* support python 3.13 and drop python 3.8 ([#1263](https://github.com/Arize-ai/openinference/issues/1263)) ([5bfaa90](https://github.com/Arize-ai/openinference/commit/5bfaa90d800a8f725b3ac7444d16972ed7821738))

## [0.1.15](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.14...python-openinference-instrumentation-dspy-v0.1.15) (2025-02-04)


### Bug Fixes

* **dspy:** dspy 2.6.0 upgrade ([#1249](https://github.com/Arize-ai/openinference/issues/1249)) ([c1ab1d8](https://github.com/Arize-ai/openinference/commit/c1ab1d86783c607c2114c92245a17ed9754ff2f4))

## [0.1.14](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.13...python-openinference-instrumentation-dspy-v0.1.14) (2024-11-12)


### Bug Fixes

* **dspy:** instrument `dspy` rather than `dspy-ai` ([#1113](https://github.com/Arize-ai/openinference/issues/1113)) ([5f6e149](https://github.com/Arize-ai/openinference/commit/5f6e149d0979a822e07f81af944c22b7530f8fed))

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.12...python-openinference-instrumentation-dspy-v0.1.13) (2024-10-10)


### Features

* **dspy:** add instrumentation for dspy adapters ([#1057](https://github.com/Arize-ai/openinference/issues/1057)) ([66799cc](https://github.com/Arize-ai/openinference/commit/66799ccf88798bf628c316276886e70ef925b9cd))


### Bug Fixes

* **dspy:** prevent the creation of duplicate span exception events ([#1058](https://github.com/Arize-ai/openinference/issues/1058)) ([54af1c3](https://github.com/Arize-ai/openinference/commit/54af1c393a03831fc908b51ca7d57ab269b13552))
* **dspy:** support dspy 2.5 and above ([#1055](https://github.com/Arize-ai/openinference/issues/1055)) ([467c8dc](https://github.com/Arize-ai/openinference/commit/467c8dcf3c58f4f443332b2062cabfe7b10de16e))
* increase version lower bound for openinference-instrumentation ([#1012](https://github.com/Arize-ai/openinference/issues/1012)) ([3236d27](https://github.com/Arize-ai/openinference/commit/3236d2733a46b84d693ddb7092209800cde8cc34))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.11...python-openinference-instrumentation-dspy-v0.1.12) (2024-08-20)


### Bug Fixes

* **dspy:** custom LM instrumentation ([#930](https://github.com/Arize-ai/openinference/issues/930)) ([7f91317](https://github.com/Arize-ai/openinference/commit/7f91317631302cb23c4b12701be2ba0b5fa3c3f0))
* **dspy:** module resolution for custom LM ([#934](https://github.com/Arize-ai/openinference/issues/934)) ([ef809be](https://github.com/Arize-ai/openinference/commit/ef809bebf4c2a19cc932b3a828cf6137be73148b))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.10...python-openinference-instrumentation-dspy-v0.1.11) (2024-08-10)


### Documentation

* TraceConfig and context attributes ([#793](https://github.com/Arize-ai/openinference/issues/793)) ([d3808c4](https://github.com/Arize-ai/openinference/commit/d3808c4bea3f6a4c72d3a7ea09b54e78072be6fd))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.9...python-openinference-instrumentation-dspy-v0.1.10) (2024-08-03)


### Features

* Enable config propagation ([#741](https://github.com/Arize-ai/openinference/issues/741)) ([16cdc6b](https://github.com/Arize-ai/openinference/commit/16cdc6b71fb14728a3eca7db27a55b68187cb4aa))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.8...python-openinference-instrumentation-dspy-v0.1.9) (2024-05-21)


### Bug Fixes

* further dspy support for non-ascii characters ([#488](https://github.com/Arize-ai/openinference/issues/488)) ([fe9d2dd](https://github.com/Arize-ai/openinference/commit/fe9d2dd453aadd6758ba3754fd4f0e68342be931))
* improve dspy support for non-ascii characters ([#478](https://github.com/Arize-ai/openinference/issues/478)) ([344bd13](https://github.com/Arize-ai/openinference/commit/344bd135ec1069d58365f25a5437cbd546b80cf0))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.7...python-openinference-instrumentation-dspy-v0.1.8) (2024-05-09)


### Features

* Make DSPY instrumentation read from attributes from context ([#421](https://github.com/Arize-ai/openinference/issues/421)) ([60c9b9c](https://github.com/Arize-ai/openinference/commit/60c9b9c2a82f4b149e1b89aff9295f9fe17fb5a7))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.6...python-openinference-instrumentation-dspy-v0.1.7) (2024-04-27)


### Bug Fixes

* **dspy:** dspy.Cohere ([#406](https://github.com/Arize-ai/openinference/issues/406)) ([d3c097d](https://github.com/Arize-ai/openinference/commit/d3c097d5356fe478f56f10451d110806f037fe10))


### Documentation

* improve dspy readme ([#297](https://github.com/Arize-ai/openinference/issues/297)) ([af34d98](https://github.com/Arize-ai/openinference/commit/af34d98d9657287625a6776ca1cab09768da6aa6))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.5...python-openinference-instrumentation-dspy-v0.1.6) (2024-03-14)


### Bug Fixes

* dspy instrumentation does not break teleprompter compilation ([#279](https://github.com/Arize-ai/openinference/issues/279)) ([f206c2e](https://github.com/Arize-ai/openinference/commit/f206c2ee4fbe0ed273e798321e5d972ddc62dac9))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.4...python-openinference-instrumentation-dspy-v0.1.5) (2024-03-13)


### Features

* add support for python 3.12 ([#271](https://github.com/Arize-ai/openinference/issues/271)) ([0556d72](https://github.com/Arize-ai/openinference/commit/0556d72997ef607545488112cde881e8660bf5db))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.3...python-openinference-instrumentation-dspy-v0.1.4) (2024-03-11)


### Features

* allow dspy instrumentation to be suppressed ([#261](https://github.com/Arize-ai/openinference/issues/261)) ([5bf584b](https://github.com/Arize-ai/openinference/commit/5bf584b32a50cb13919797c4fcebd7ccb5606ec7))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.2...python-openinference-instrumentation-dspy-v0.1.3) (2024-03-06)


### Bug Fixes

* add support for DSPy's Google model class ([#255](https://github.com/Arize-ai/openinference/issues/255)) ([8354956](https://github.com/Arize-ai/openinference/commit/83549561d1749eed4ab74c423fadec2c934935ca))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.1...python-openinference-instrumentation-dspy-v0.1.2) (2024-03-04)


### Bug Fixes

* support dpsy 2.3.4 and above ([#250](https://github.com/Arize-ai/openinference/issues/250)) ([e036418](https://github.com/Arize-ai/openinference/commit/e0364189281d5f0073000f8ff02ac4daef321c4e))


### Documentation

* Convert READMEs to Markdown ([#227](https://github.com/Arize-ai/openinference/issues/227)) ([e4bcf4b](https://github.com/Arize-ai/openinference/commit/e4bcf4b86f27cc119a77f551811f9142ec6075ce))

## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-dspy-v0.1.0...python-openinference-instrumentation-dspy-v0.1.1) (2024-02-01)


### Features

* dspy instrumentation enhancements ([#158](https://github.com/Arize-ai/openinference/issues/158)) ([ff828bb](https://github.com/Arize-ai/openinference/commit/ff828bba1b2aec118401855eea2cd0c2f72af4a2))

## 0.1.0 (2024-01-24)


### Features

* **python:** dspy lm instrumentation ([#126](https://github.com/Arize-ai/openinference/issues/126)) ([92714bc](https://github.com/Arize-ai/openinference/commit/92714bcc942d516211003c75f36acba413c06858))
