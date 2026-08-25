# @arizeai/openinference-instrumentation-claude-agent-sdk

## 0.2.14

### Patch Changes

- Updated dependencies [1fe497f]
  - @arizeai/openinference-semantic-conventions@2.8.0
  - @arizeai/openinference-core@2.5.3

## 0.2.13

### Patch Changes

- 74ae809: Replace unsafe type assertions with runtime type guards across packages (enforce `typescript/no-unsafe-type-assertion`)
- Updated dependencies [74ae809]
  - @arizeai/openinference-core@2.5.2

## 0.2.12

### Patch Changes

- Updated dependencies [237ce2b]
  - @arizeai/openinference-semantic-conventions@2.7.0
  - @arizeai/openinference-core@2.5.1

## 0.2.11

### Patch Changes

- Updated dependencies [0168198]
  - @arizeai/openinference-core@2.5.0

## 0.2.10

### Patch Changes

- Updated dependencies [145e3c6]
  - @arizeai/openinference-semantic-conventions@2.6.0
  - @arizeai/openinference-core@2.4.1

## 0.2.9

### Patch Changes

- 2819fcb: Fix native ESM instrumentation for Claude Agent SDK module namespaces whose exports cannot be reassigned.

## 0.2.8

### Patch Changes

- Updated dependencies [d0f5a88]
  - @arizeai/openinference-core@2.4.0

## 0.2.7

### Patch Changes

- Updated dependencies [1fe7927]
  - @arizeai/openinference-core@2.3.0

## 0.2.6

### Patch Changes

- Updated dependencies [0f0242c]
- Updated dependencies [26733d8]
  - @arizeai/openinference-semantic-conventions@2.5.0
  - @arizeai/openinference-core@2.2.0

## 0.2.5

### Patch Changes

- Updated dependencies [81b8bdb]
  - @arizeai/openinference-semantic-conventions@2.4.0
  - @arizeai/openinference-core@2.1.1

## 0.2.4

### Patch Changes

- Updated dependencies [cfb128c]
  - @arizeai/openinference-core@2.1.0

## 0.2.3

### Patch Changes

- Updated dependencies [e09ce3f]
  - @arizeai/openinference-semantic-conventions@2.3.0
  - @arizeai/openinference-core@2.0.8

## 0.2.2

### Patch Changes

- Updated dependencies [4eebba3]
  - @arizeai/openinference-core@2.0.7

## 0.2.1

### Patch Changes

- Updated dependencies [7eb1c88]
- Updated dependencies [3944459]
  - @arizeai/openinference-semantic-conventions@2.2.0
  - @arizeai/openinference-core@2.0.6

## 0.2.0

### Minor Changes

- cacb415: Initial release of OpenInference instrumentation for Claude Agent SDK

  - Instruments V1 (`query()`) and V2 (`unstable_v2_prompt`, `unstable_v2_createSession`, `unstable_v2_resumeSession`) APIs
  - Produces AGENT spans for query/prompt/session-turn lifecycles
  - Produces TOOL child spans via hook injection (PreToolUse/PostToolUse/PostToolUseFailure)
  - Captures input/output values, token counts, cost, session ID, and model name
  - Supports trace configuration for masking sensitive data
  - Supports both CommonJS and ESM module loading
