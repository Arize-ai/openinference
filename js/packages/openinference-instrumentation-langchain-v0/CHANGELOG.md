# @arizeai/openinference-instrumentation-langchain-v0

## 0.0.12

### Patch Changes

- 0f0242c: Add `PROMPT` to the `OpenInferenceSpanKind` enum, aligning the JS package with the OpenInference spec and the Python semantic conventions. LangChain prompt template spans now correctly report `openinference.span.kind = "PROMPT"` instead of falling through to `"CHAIN"`.
- Updated dependencies [0f0242c]
- Updated dependencies [26733d8]
  - @arizeai/openinference-semantic-conventions@2.5.0
  - @arizeai/openinference-core@2.2.0

## 0.0.11

### Patch Changes

- Updated dependencies [81b8bdb]
  - @arizeai/openinference-semantic-conventions@2.4.0
  - @arizeai/openinference-core@2.1.1

## 0.0.10

### Patch Changes

- Updated dependencies [cfb128c]
  - @arizeai/openinference-core@2.1.0

## 0.0.9

### Patch Changes

- Updated dependencies [e09ce3f]
  - @arizeai/openinference-semantic-conventions@2.3.0
  - @arizeai/openinference-core@2.0.8

## 0.0.8

### Patch Changes

- Updated dependencies [4eebba3]
  - @arizeai/openinference-core@2.0.7

## 0.0.7

### Patch Changes

- Updated dependencies [7eb1c88]
- Updated dependencies [3944459]
  - @arizeai/openinference-semantic-conventions@2.2.0
  - @arizeai/openinference-core@2.0.6

## 0.0.6

### Patch Changes

- c79c564: force publish
- c79c564: signed publishing
- Updated dependencies [c79c564]
- Updated dependencies [c79c564]
  - @arizeai/openinference-core@2.0.5
  - @arizeai/openinference-semantic-conventions@2.1.7

## 0.0.5

### Patch Changes

- a4eead1: force publish
- a4eead1: signed publishing
- Updated dependencies [a4eead1]
- Updated dependencies [a4eead1]
  - @arizeai/openinference-core@2.0.4
  - @arizeai/openinference-semantic-conventions@2.1.6

## 0.0.4

### Patch Changes

- 74f278c: force publish
- 74f278c: signed publishing
- Updated dependencies [74f278c]
- Updated dependencies [74f278c]
  - @arizeai/openinference-core@2.0.3
  - @arizeai/openinference-semantic-conventions@2.1.5

## 0.0.3

### Patch Changes

- fe61379: force publish
- fe61379: signed publishing
- Updated dependencies [fe61379]
- Updated dependencies [fe61379]
  - @arizeai/openinference-core@2.0.2
  - @arizeai/openinference-semantic-conventions@2.1.4

## 0.0.2

### Patch Changes

- 006a685: signed publishing
- Updated dependencies [006a685]
  - @arizeai/openinference-core@2.0.1
  - @arizeai/openinference-semantic-conventions@2.1.3

## 0.0.1

### Patch Changes

- 15c18b0: Initial publish for langchain 0.X. The main instrumentor will be moving to 1.X
