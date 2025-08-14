# @arizeai/openinference-instrumentation-openai

## 3.2.2

### Patch Changes

- Updated dependencies [9d3bdb4]
  - @arizeai/openinference-core@1.0.6

## 3.2.1

### Patch Changes

- Updated dependencies [59be946]
  - @arizeai/openinference-semantic-conventions@2.1.1
  - @arizeai/openinference-core@1.0.5

## 3.2.0

### Minor Changes

- e211094: add ability to use a non-global trace provider

## 3.1.1

### Patch Changes

- Updated dependencies [34a4159]
  - @arizeai/openinference-semantic-conventions@2.1.0
  - @arizeai/openinference-core@1.0.4

## 3.1.0

### Minor Changes

- c9b96a7: feat: Add support for responses.parse and chat.parse method instrumentation

## 3.0.0

### Major Changes

- 35f7b0e: feat: Add support for openai-node sdk 5.x

  Support for openai@4.x has been dropped. Please upgrade to openai@5.x to continue using this package.

## 2.3.1

### Patch Changes

- Updated dependencies [c2ee804]
- Updated dependencies [5f904bf]
- Updated dependencies [5f90a80]
  - @arizeai/openinference-semantic-conventions@2.0.0
  - @arizeai/openinference-core@1.0.3

## 2.3.0

### Minor Changes

- 5207deb: feat(openai-js): Instrument OpenAI responses sdk

## 2.2.0

### Minor Changes

- e5300f3: support for prompt and completion token count details for langchainjs and openai

## 2.1.0

### Minor Changes

- 5aa6511: fix(openai): Rewrite namespace import to default import in support of ESM projects

### Patch Changes

- Updated dependencies [ae5cd15]
  - @arizeai/openinference-semantic-conventions@1.1.0
  - @arizeai/openinference-core@1.0.2

## 2.0.3

### Patch Changes

- Updated dependencies [c4e2252]
  - @arizeai/openinference-semantic-conventions@1.0.1
  - @arizeai/openinference-core@1.0.1

## 2.0.2

### Patch Changes

- f1c1a3c: place image attributes under the right path

## 2.0.1

### Patch Changes

- b69cca6: Account for developer role in the openai instrumentation

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

## 1.1.2

### Patch Changes

- Updated dependencies [1188c6d]
  - @arizeai/openinference-semantic-conventions@0.14.0
  - @arizeai/openinference-core@0.3.3

## 1.1.1

### Patch Changes

- Updated dependencies [710d1d3]
  - @arizeai/openinference-semantic-conventions@0.13.0
  - @arizeai/openinference-core@0.3.2

## 1.1.0

### Minor Changes

- a0e6f30: Support tool_call_id and tool_call.id

### Patch Changes

- 47cd6b5: fix: propgate context to spans created as a result of work done within openai calls
- Updated dependencies [a0e6f30]
  - @arizeai/openinference-semantic-conventions@0.12.0
  - @arizeai/openinference-core@0.3.1

## 1.0.0

### Major Changes

- c03a5b6: add support for trace config to OpenAI and LangChain auto instrumentors to allow for attribute masking on spans

## 0.6.0

### Minor Changes

- f965410: Add system and provider attributes to openai spans

### Patch Changes

- Updated dependencies [f965410]
- Updated dependencies [712b9da]
- Updated dependencies [d200d85]
  - @arizeai/openinference-semantic-conventions@0.11.0
  - @arizeai/openinference-core@0.3.0

## 0.5.0

### Minor Changes

- 32968be: Add tool call schema to openAI instrumentation

### Patch Changes

- Updated dependencies [3b8702a]
  - @arizeai/openinference-core@0.2.0

## 0.4.1

### Patch Changes

- Updated dependencies [ba142d5]
  - @arizeai/openinference-semantic-conventions@0.10.0

## 0.4.0

### Minor Changes

- 5381ec9: capture images in request

### Patch Changes

- Updated dependencies [28a4ea2]
- Updated dependencies [96af3d6]
  - @arizeai/openinference-semantic-conventions@0.9.0

## 0.3.4

### Patch Changes

- Updated dependencies [9affdf6]
  - @arizeai/openinference-semantic-conventions@0.8.0

## 0.3.3

### Patch Changes

- Updated dependencies [9c44b14]
  - @arizeai/openinference-semantic-conventions@0.7.0

## 0.3.2

### Patch Changes

- Updated dependencies [b66bf54]
- Updated dependencies [fe69250]
- Updated dependencies [60ade67]
  - @arizeai/openinference-semantic-conventions@0.6.0

## 0.3.1

### Patch Changes

- Updated dependencies [0d1d065]
  - @arizeai/openinference-semantic-conventions@0.5.0

## 0.3.0

### Minor Changes

- ab5504f: Add support for patch / unpatch of packages that are made immutable (Deno, webpack)

## 0.2.0

### Minor Changes

- 53311f5: Add manual instrumentation for openai node sdk

## 0.1.4

### Patch Changes

- Updated dependencies [921a40c]
  - @arizeai/openinference-semantic-conventions@0.4.0

## 0.1.3

### Patch Changes

- Updated dependencies [9af89e5]
  - @arizeai/openinference-semantic-conventions@0.3.0

## 0.1.2

### Patch Changes

- Updated dependencies [04c303f]
  - @arizeai/openinference-semantic-conventions@0.2.0

## 0.1.1

### Patch Changes

- 68c92d3: Support tool and function calls while streaming

## 0.1.0

### Minor Changes

- 1925aad: add support for legacy completions api
- 82c5d83: Add streaming instrumentation for OpenAI Chat completion

### Patch Changes

- Updated dependencies [1925aad]
  - @arizeai/openinference-semantic-conventions@0.1.0

## 0.0.4

### Patch Changes

- 9b3bc4a: Add OpenAI Embeddings sementic attributes and instrumentation
- Updated dependencies [9b3bc4a]
  - @arizeai/openinference-semantic-conventions@0.0.13

## 0.0.3

### Patch Changes

- Updated dependencies
  - @arizeai/openinference-semantic-conventions@0.0.12

## 0.0.2

### Patch Changes

- Token count support for OpenAI
- OpenAI instrumentation
- Updated dependencies
- Updated dependencies
  - @arizeai/openinference-semantic-conventions@0.0.11

## 0.0.1

### Patch Changes

- Updated dependencies
  - @arizeai/openinference-semantic-conventions@0.0.9

## 0.0.1

### Patch Changes

- Updated dependencies [49c3e71]
  - @arizeai/openinference-semantic-conventions@0.0.8
