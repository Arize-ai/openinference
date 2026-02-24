# @arizeai/openinference-vercel

## 2.7.0

### Minor Changes

- 2821f95: feat(openinference-vercel): Expand multi-tool results into multiple tool messages

## 2.6.0

### Minor Changes

- 912cdbe: feat(openinference-vercel): add AI SDK v6 telemetry support

  This release improves compatibility with AI SDK v6 telemetry while keeping best-effort compatibility with older AI SDK versions.

  Key behavior:
  - Prefer standard `gen_ai.*` attributes (OTel GenAI semantic conventions) when present
  - Fall back to Vercel-specific `ai.*` attributes for data not available in `gen_ai.*` and for older SDK versions

  Vercel-specific `ai.*` processing includes:
  - Span kind determination from `operation.name`
  - Embeddings (`ai.value`, `ai.embedding`, etc.)
  - Tool calls (`ai.toolCall.*`)
  - Metadata (`ai.telemetry.metadata.*`)
  - Streaming metrics (`ai.response.msToFirstChunk`, etc.)
  - Input/output messages from `ai.prompt.messages` and `ai.response.toolCalls`

  Additional improvements:
  - Root AI SDK spans now have a status set (`OK`/`ERROR`) based on the overall invocation result.

  Notes:
  - AI SDK telemetry is experimental; older versions are supported on a best-effort basis.

  **Migration Guide:**
  - If you are on AI SDK v6: no code changes required.
  - If you are on older AI SDK versions: no code changes required; compatibility is best-effort.

## 2.5.5

### Patch Changes

- c79c564: force publish
- c79c564: signed publishing
- Updated dependencies [c79c564]
- Updated dependencies [c79c564]
  - @arizeai/openinference-core@2.0.5
  - @arizeai/openinference-semantic-conventions@2.1.7

## 2.5.4

### Patch Changes

- a4eead1: force publish
- a4eead1: signed publishing
- Updated dependencies [a4eead1]
- Updated dependencies [a4eead1]
  - @arizeai/openinference-core@2.0.4
  - @arizeai/openinference-semantic-conventions@2.1.6

## 2.5.3

### Patch Changes

- 74f278c: force publish
- 74f278c: signed publishing
- Updated dependencies [74f278c]
- Updated dependencies [74f278c]
  - @arizeai/openinference-core@2.0.3
  - @arizeai/openinference-semantic-conventions@2.1.5

## 2.5.2

### Patch Changes

- fe61379: force publish
- fe61379: signed publishing
- Updated dependencies [fe61379]
- Updated dependencies [fe61379]
  - @arizeai/openinference-core@2.0.2
  - @arizeai/openinference-semantic-conventions@2.1.4

## 2.5.1

### Patch Changes

- 006a685: signed publishing
- Updated dependencies [006a685]
  - @arizeai/openinference-core@2.0.1
  - @arizeai/openinference-semantic-conventions@2.1.3

## 2.5.0

### Minor Changes

- 95f4c5f: feat: Trace new token usage keys in ai sdk v5

## 2.4.0

### Minor Changes

- 6103271: # feat: Add support for ai sdk v5 tools

### Patch Changes

- Updated dependencies [d3d7017]
  - @arizeai/openinference-core@2.0.0

## 2.3.5

### Patch Changes

- Updated dependencies [5161c9f]
  - @arizeai/openinference-core@1.0.8

## 2.3.4

### Patch Changes

- Updated dependencies [c50ffb0]
  - @arizeai/openinference-semantic-conventions@2.1.2
  - @arizeai/openinference-core@1.0.7

## 2.3.3

### Patch Changes

- Updated dependencies [9d3bdb4]
  - @arizeai/openinference-core@1.0.6

## 2.3.2

### Patch Changes

- Updated dependencies [59be946]
  - @arizeai/openinference-semantic-conventions@2.1.1
  - @arizeai/openinference-core@1.0.5

## 2.3.1

### Patch Changes

- fc7f97b: do not override existing span kind on a span

## 2.3.0

### Minor Changes

- aaed014: fix: Increase opentelemetry/api peer dependency ranges for compatibility with vercel ai

## 2.2.2

### Patch Changes

- Updated dependencies [34a4159]
  - @arizeai/openinference-semantic-conventions@2.1.0
  - @arizeai/openinference-core@1.0.4

## 2.2.1

### Patch Changes

- Updated dependencies [c2ee804]
- Updated dependencies [5f904bf]
- Updated dependencies [5f90a80]
  - @arizeai/openinference-semantic-conventions@2.0.0
  - @arizeai/openinference-core@1.0.3

## 2.2.0

### Minor Changes

- a573489: feat: Mastra instrumentation

  Initial instrumentation for Mastra, adhering to OpenInference semantic conventions

- a573489: feat: Instrument tool calls and results from multi-part content messages

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
