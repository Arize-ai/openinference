# @arizeai/openinference-vercel

## 2.1.0

### Minor Changes

- c301f99: chore: pin peer deps, update readme, fix types

## 2.0.3

### Patch Changes

- Updated dependencies [ae5cd15]
  - @arizeai/openinference-semantic-conventions@1.1.0
  - @arizeai/openinference-core@1.0.2

## 2.0.2

### Patch Changes

- Updated dependencies [c4e2252]
  - @arizeai/openinference-semantic-conventions@1.0.1
  - @arizeai/openinference-core@1.0.1

## 2.0.1

### Patch Changes

- 365a3c2: Updated the OpenInference semantic convention mapping to account for changes to the Vercel AI SDK semantic conventions

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

## 1.2.2

### Patch Changes

- Updated dependencies [1188c6d]
  - @arizeai/openinference-semantic-conventions@0.14.0
  - @arizeai/openinference-core@0.3.3

## 1.2.1

### Patch Changes

- Updated dependencies [710d1d3]
  - @arizeai/openinference-semantic-conventions@0.13.0
  - @arizeai/openinference-core@0.3.2

## 1.2.0

### Minor Changes

- a0e6f30: Support tool_call_id and tool_call.id

### Patch Changes

- Updated dependencies [a0e6f30]
  - @arizeai/openinference-semantic-conventions@0.12.0
  - @arizeai/openinference-core@0.3.1

## 1.1.0

### Minor Changes

- a96fbd5: Add readme documentation

### Patch Changes

- Updated dependencies [f965410]
- Updated dependencies [712b9da]
- Updated dependencies [d200d85]
  - @arizeai/openinference-semantic-conventions@0.11.0
  - @arizeai/openinference-core@0.3.0

## 1.0.0

### Major Changes

- 4f9246f: migrate OpenInferenceSpanProcessor to OpenInferenceSimpleSpanProcessor and OpenInferenceBatchSpanProcessor to allow for filtering exported spans

## 0.1.1

### Patch Changes

- 3b8702a: remove generic log from withSafety and add onError callback
- ff2668c: caputre input and output for tools, fix double count of tokens on llm spans / chains
- Updated dependencies [3b8702a]
  - @arizeai/openinference-core@0.2.0

## 0.1.0

### Minor Changes

- 97ca03b: add OpenInferenceSpanProcessor to transform Vercel AI SDK Spans to conform to the OpenInference spec

### Patch Changes

- Updated dependencies [ba142d5]
  - @arizeai/openinference-semantic-conventions@0.10.0
  - @arizeai/openinference-core@0.1.1
