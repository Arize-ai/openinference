# @arizeai/openinference-core

## 1.0.0

### Major Changes

- 16a3815: ESM support

  Packages are now shipped as "Dual Package" meaning that ESM and CJS module resolution
  should be supported for each package.

  Support is described as "experimental" because opentelemetry describes support for autoinstrumenting
  ESM projects as "ongoing". See https://github.com/open-telemetry/opentelemetry-js/blob/61d5a0e291db26c2af638274947081b29db3f0ca/doc/esm-support.md

### Patch Changes

- Updated dependencies [16a3815]
  - @arizeai/openinference-semantic-conventions@1.0.0

## 0.3.3

### Patch Changes

- Updated dependencies [1188c6d]
  - @arizeai/openinference-semantic-conventions@0.14.0

## 0.3.2

### Patch Changes

- Updated dependencies [710d1d3]
  - @arizeai/openinference-semantic-conventions@0.13.0

## 0.3.1

### Patch Changes

- Updated dependencies [a0e6f30]
  - @arizeai/openinference-semantic-conventions@0.12.0

## 0.3.0

### Minor Changes

- 712b9da: add OITracer and trace config to allow for masking of sensitive information on spans

### Patch Changes

- Updated dependencies [f965410]
- Updated dependencies [d200d85]
  - @arizeai/openinference-semantic-conventions@0.11.0

## 0.2.0

### Minor Changes

- 3b8702a: remove generic log from withSafety and add onError callback

## 0.1.1

### Patch Changes

- Updated dependencies [ba142d5]
  - @arizeai/openinference-semantic-conventions@0.10.0

## 0.1.0

### Minor Changes

- 92f7fb1: Adds support for context attributes which can be propagated to all spans within the context scope

### Patch Changes

- 3d00a02: removes isAttributeValue to pull from open telemetry, exports utilities
- Updated dependencies [28a4ea2]
- Updated dependencies [96af3d6]
  - @arizeai/openinference-semantic-conventions@0.9.0
