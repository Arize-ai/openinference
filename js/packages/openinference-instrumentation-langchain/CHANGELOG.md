# @arizeai/openinference-instrumentation-langchain

## 3.3.0

### Minor Changes

- 2869d93: captures the json schema for function calls that are bound to llm calls as tools

## 3.2.0

### Minor Changes

- 6892184: feat(langchain-js): Instrument tool messages in langchain js instrumentation

### Patch Changes

- 4b296d2: properly record exceptions on spans for langchain runs

## 3.1.0

### Minor Changes

- e5300f3: support for prompt and completion token count details for langchainjs and openai

## 3.0.2

### Patch Changes

- Updated dependencies [ae5cd15]
  - @arizeai/openinference-semantic-conventions@1.1.0
  - @arizeai/openinference-core@1.0.2

## 3.0.1

### Patch Changes

- Updated dependencies [c4e2252]
  - @arizeai/openinference-semantic-conventions@1.0.1
  - @arizeai/openinference-core@1.0.1

## 3.0.0

### Major Changes

- 1ae380c: deprecate support for v0.1

## 2.0.0

### Major Changes

- 16a3815: ESM support

  Packages are now shipped as "Dual Package" meaning that ESM and CJS module resolution
  should be supported for each package.

  Support is described as "experimental" because opentelemetry describes support for autoinstrumenting
  ESM projects as "ongoing". See https://github.com/open-telemetry/opentelemetry-js/blob/61d5a0e291db26c2af638274947081b29db3f0ca/doc/esm-support.md

### Patch Changes

- Updated dependencies [16a3815]
  - @arizeai/openinference-semantic-conventions@1.0.0
  - @arizeai/openinference-core@1.0.0

## 1.1.0

### Minor Changes

- 8182c3a: Parses metadata with key name in (session_id, thread_id, conversation_id) into semantic conventions session id

### Patch Changes

- Updated dependencies [1188c6d]
  - @arizeai/openinference-semantic-conventions@0.14.0
  - @arizeai/openinference-core@0.3.3

## 1.0.2

### Patch Changes

- Updated dependencies [710d1d3]
  - @arizeai/openinference-semantic-conventions@0.13.0
  - @arizeai/openinference-core@0.3.2

## 1.0.1

### Patch Changes

- 210ab8c: fix: add support for capturing ChatBedrock token counts
- Updated dependencies [a0e6f30]
  - @arizeai/openinference-semantic-conventions@0.12.0
  - @arizeai/openinference-core@0.3.1

## 1.0.0

### Major Changes

- c03a5b6: add support for trace config to OpenAI and LangChain auto instrumentors to allow for attribute masking on spans

### Minor Changes

- 518f298: add support for @langchain/core version ^0.3.0

## 0.2.1

### Patch Changes

- Updated dependencies [f965410]
- Updated dependencies [712b9da]
- Updated dependencies [d200d85]
  - @arizeai/openinference-semantic-conventions@0.11.0
  - @arizeai/openinference-core@0.3.0

## 0.2.0

### Minor Changes

- 14ac4ae: add support for langchain version ^0.2

### Patch Changes

- Updated dependencies [3b8702a]
  - @arizeai/openinference-core@0.2.0

## 0.1.1

### Patch Changes

- Updated dependencies [ba142d5]
  - @arizeai/openinference-semantic-conventions@0.10.0

## 0.1.0

### Minor Changes

- af1184e: Fix the way that instrumentation stores whether or not it is patched by storing patched state in the closure"

### Patch Changes

- Updated dependencies [28a4ea2]
- Updated dependencies [96af3d6]
  - @arizeai/openinference-semantic-conventions@0.9.0

## 0.0.9

### Patch Changes

- Updated dependencies [9affdf6]
  - @arizeai/openinference-semantic-conventions@0.8.0

## 0.0.8

### Patch Changes

- Updated dependencies [9c44b14]
  - @arizeai/openinference-semantic-conventions@0.7.0

## 0.0.7

### Patch Changes

- Updated dependencies [b66bf54]
- Updated dependencies [fe69250]
- Updated dependencies [60ade67]
  - @arizeai/openinference-semantic-conventions@0.6.0

## 0.0.6

### Patch Changes

- Updated dependencies [0d1d065]
  - @arizeai/openinference-semantic-conventions@0.5.0

## 0.0.5

### Patch Changes

- Updated dependencies [921a40c]
  - @arizeai/openinference-semantic-conventions@0.4.0

## 0.0.4

### Patch Changes

- ecda94a: update README to include manual instrumentation step

## 0.0.3

### Patch Changes

- 5d21aac: update fallback span kind to be chain

## 0.0.2

### Patch Changes

- a3c52e0: add manuallyInstrument method to LangChainInsturmentation

## 0.0.1

### Patch Changes

- cf56c6f: add LangChain instrumentation
