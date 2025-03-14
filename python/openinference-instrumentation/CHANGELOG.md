# Changelog

## [0.1.24](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.23...python-openinference-instrumentation-v0.1.24) (2025-03-14)


### Features

* openai-agents instrumentation ([#1350](https://github.com/Arize-ai/openinference/issues/1350)) ([9afbad3](https://github.com/Arize-ai/openinference/commit/9afbad3100d68601a2f9265fe20985a34f80e04b))


### Documentation

* fix license to be openinference ([#1353](https://github.com/Arize-ai/openinference/issues/1353)) ([85d435b](https://github.com/Arize-ai/openinference/commit/85d435be3af3de5424494cfbdd654454688b7377))

## [0.1.23](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.22...python-openinference-instrumentation-v0.1.23) (2025-02-27)


### Bug Fixes

* allow user override of id generator ([#1315](https://github.com/Arize-ai/openinference/issues/1315)) ([1916749](https://github.com/Arize-ai/openinference/commit/19167498fd74f2e93481bd63b5636e264af1eaab))

## [0.1.22](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.21...python-openinference-instrumentation-v0.1.22) (2025-02-04)


### Bug Fixes

* support python 3.13 and drop python 3.8 ([#1263](https://github.com/Arize-ai/openinference/issues/1263)) ([5bfaa90](https://github.com/Arize-ai/openinference/commit/5bfaa90d800a8f725b3ac7444d16972ed7821738))

## [0.1.21](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.20...python-openinference-instrumentation-v0.1.21) (2025-01-23)


### Features

* openinference tracer ([#1147](https://github.com/Arize-ai/openinference/issues/1147)) ([22d80ca](https://github.com/Arize-ai/openinference/commit/22d80ca066a8d29e9b9ef08ce581b4a7ad4eb08b))

## [0.1.20](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.19...python-openinference-instrumentation-v0.1.20) (2024-12-13)


### Features

* add ability to hide LLM invocation paramaters for OITracer ([#1171](https://github.com/Arize-ai/openinference/issues/1171)) ([f7e94c7](https://github.com/Arize-ai/openinference/commit/f7e94c7f658570169c564f11663fc9eeaee05f46))

## [0.1.19](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.18...python-openinference-instrumentation-v0.1.19) (2024-12-02)


### Bug Fixes

* increase python upperbound to include 3.13 for openinference-instrumentation ([#1137](https://github.com/Arize-ai/openinference/issues/1137)) ([0c2f297](https://github.com/Arize-ai/openinference/commit/0c2f297bb479b6cd4a70c7e0b28d6578e0abc6e3))

## [0.1.18](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.17...python-openinference-instrumentation-v0.1.18) (2024-09-11)


### Bug Fixes

* add missing dependency opentelemetry-sdk ([#1026](https://github.com/Arize-ai/openinference/issues/1026)) ([11e6cb9](https://github.com/Arize-ai/openinference/commit/11e6cb98cf2efe73fd3b3972869d8527db67cc72))

## [0.1.17](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.16...python-openinference-instrumentation-v0.1.17) (2024-09-10)


### Features

* id generator with separate source of randomness ([#1010](https://github.com/Arize-ai/openinference/issues/1010)) ([ac8cce1](https://github.com/Arize-ai/openinference/commit/ac8cce112341bb31a575cb1e61a55acb196fc600))

## [0.1.16](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.15...python-openinference-instrumentation-v0.1.16) (2024-09-04)


### Bug Fixes

* remove setting the global logger level ([#1001](https://github.com/Arize-ai/openinference/issues/1001)) ([5c0f83c](https://github.com/Arize-ai/openinference/commit/5c0f83c355304da289ae1c849b9d315990281184))

## [0.1.15](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.14...python-openinference-instrumentation-v0.1.15) (2024-08-16)


### Features

* attribute prioritization ([#906](https://github.com/Arize-ai/openinference/issues/906)) ([0add042](https://github.com/Arize-ai/openinference/commit/0add0421b5f0d9b64c579027c469513359863a68))

## [0.1.14](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.13...python-openinference-instrumentation-v0.1.14) (2024-08-15)


### Bug Fixes

* mask attributes when starting span ([#892](https://github.com/Arize-ai/openinference/issues/892)) ([9b72287](https://github.com/Arize-ai/openinference/commit/9b72287401d5c424a8951e1d6a15cca14fcd05cc))

## [0.1.13](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.12...python-openinference-instrumentation-v0.1.13) (2024-08-10)


### Documentation

* TraceConfig and context attributes ([#793](https://github.com/Arize-ai/openinference/issues/793)) ([d3808c4](https://github.com/Arize-ai/openinference/commit/d3808c4bea3f6a4c72d3a7ea09b54e78072be6fd))

## [0.1.12](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.11...python-openinference-instrumentation-v0.1.12) (2024-08-01)


### Bug Fixes

* Rename base tracer and masked span ([#693](https://github.com/Arize-ai/openinference/issues/693)) ([861ea4b](https://github.com/Arize-ai/openinference/commit/861ea4ba45cf02a1d0519a7cd2c5c6ca5d74115b))

## [0.1.11](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.10...python-openinference-instrumentation-v0.1.11) (2024-07-31)


### Bug Fixes

* Ensure string type to check URLs ([#682](https://github.com/Arize-ai/openinference/issues/682)) ([24c51ec](https://github.com/Arize-ai/openinference/commit/24c51ece2b50a36c791f9e0c72088360fe91ca5f))

## [0.1.10](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.9...python-openinference-instrumentation-v0.1.10) (2024-07-31)


### Features

* Config settings concentrated in common instrumentation package ([#673](https://github.com/Arize-ai/openinference/issues/673)) ([3e34897](https://github.com/Arize-ai/openinference/commit/3e348979e9db5a73ba7f8edac49e1c01816d89e7))

## [0.1.9](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.8...python-openinference-instrumentation-v0.1.9) (2024-07-29)


### Features

* Add TracingConfig for configuration settings in instrumentations ([#639](https://github.com/Arize-ai/openinference/issues/639)) ([fcea7f9](https://github.com/Arize-ai/openinference/commit/fcea7f99e505f104543d3a51a9b3b0f25cba8072))

## [0.1.8](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.7...python-openinference-instrumentation-v0.1.8) (2024-07-09)


### Features

* helper functions to convert OTEL IDs for spans and traces to strings ([#579](https://github.com/Arize-ai/openinference/issues/579)) ([dc60a92](https://github.com/Arize-ai/openinference/commit/dc60a92f8690243b5277cfba4c7e68a2056e293f))

## [0.1.7](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.6...python-openinference-instrumentation-v0.1.7) (2024-05-21)


### Bug Fixes

* improve handling of non-ASCII unicode characters in openinference-instrumentation ([#473](https://github.com/Arize-ai/openinference/issues/473)) ([f8236a4](https://github.com/Arize-ai/openinference/commit/f8236a49f88aaaf0ffec2f0d7a06ce42ba3814d7))

## [0.1.6](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.5...python-openinference-instrumentation-v0.1.6) (2024-05-14)


### Bug Fixes

* ensure bazel can discover instrumentation modules in openinference.instrumentations ([#455](https://github.com/Arize-ai/openinference/issues/455)) ([38a3ad6](https://github.com/Arize-ai/openinference/commit/38a3ad6cca3a931ebbe51a57bc78c4a000dcae17))

## [0.1.5](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.4...python-openinference-instrumentation-v0.1.5) (2024-05-06)


### Bug Fixes

* Add missing context variables to list ([#438](https://github.com/Arize-ai/openinference/issues/438)) ([ab2ef46](https://github.com/Arize-ai/openinference/commit/ab2ef4655c16e662c499b0302f4a0b28892f6b6c))

## [0.1.4](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.3...python-openinference-instrumentation-v0.1.4) (2024-05-06)


### Features

* Add context manager for prompt template, version, and variables ([#437](https://github.com/Arize-ai/openinference/issues/437)) ([0d44490](https://github.com/Arize-ai/openinference/commit/0d444904afa13f33c67a7e520eeb819fc7208ddf))
* Allow context managers as decorators ([#431](https://github.com/Arize-ai/openinference/issues/431)) ([b1bb379](https://github.com/Arize-ai/openinference/commit/b1bb379bad97f811668dcc6d8c37760944bf03ff))

## [0.1.3](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.2...python-openinference-instrumentation-v0.1.3) (2024-05-03)


### Bug Fixes

* require minimum version of openinference-semantic-conventions ([#424](https://github.com/Arize-ai/openinference/issues/424)) ([040c1aa](https://github.com/Arize-ai/openinference/commit/040c1aa54a36d7312097938d87b187536d87e20a))

## [0.1.2](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.1...python-openinference-instrumentation-v0.1.2) (2024-05-02)


### Features

* Create context managers to pass session, user, metadata and tags as span attributes ([#414](https://github.com/Arize-ai/openinference/issues/414)) ([c142b3a](https://github.com/Arize-ai/openinference/commit/c142b3a1adcb286d9d4be00d7bbe34c23f6e6805))


## [0.1.1](https://github.com/Arize-ai/openinference/compare/python-openinference-instrumentation-v0.1.0...python-openinference-instrumentation-v0.1.1) (2024-03-26)


### Bug Fixes

* add license file for openinference-instrumentation ([#344](https://github.com/Arize-ai/openinference/issues/344)) ([204bfb8](https://github.com/Arize-ai/openinference/commit/204bfb8b4179d06e72ad76f6e676028f4527a8ae))

## 0.1.0 (2024-03-26)


### Features

* add openinference-instrumentation package ([#340](https://github.com/Arize-ai/openinference/issues/340)) ([1e895c8](https://github.com/Arize-ai/openinference/commit/1e895c800feddf08f08babc34eabad9d9429ee51))
