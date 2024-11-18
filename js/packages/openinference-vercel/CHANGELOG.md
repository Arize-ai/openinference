# @arizeai/openinference-vercel

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
