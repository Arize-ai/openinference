# @arizeai/openinference-instrumentation-typesafe

## 0.3.1

### Patch Changes

- Updated dependencies [0d26a59]
  - @arizeai/openinference-core@2.8.0

## 0.3.0

### Minor Changes

- 565edea: Record `TypeSafeClient.systemOne` calls as `DECISION` spans instead of `LLM` spans. System One scores or selects among the candidate options supplied in the request rather than generating free-form text, which is what the new `DECISION` span kind describes. The `llm.*` attributes (provider, system, model names, invocation parameters, token counts) are unchanged.

### Patch Changes

- Updated dependencies [53b7a0e]
  - @arizeai/openinference-semantic-conventions@2.14.0
  - @arizeai/openinference-core@2.7.3

## 0.2.1

### Patch Changes

- Updated dependencies [a1f276c]
- Updated dependencies [a719562]
  - @arizeai/openinference-semantic-conventions@2.13.0
  - @arizeai/openinference-core@2.7.2

## 0.2.0

### Minor Changes

- 5a075b9: Add TypeSafe AI SDK instrumentation with one LLM span per systemOne call, structured JSON input/output payloads, question confidence metadata, token usage, context propagation, and configurable masking. Preserve the SDK's APIPromise interface and support both ESM and CommonJS. Add TypeSafe provider and system values to the semantic conventions and recognize the TypeSafe API hostname in provider inference.

### Patch Changes

- 32ce597: Test release to verify the publish pipeline for the TypeSafe instrumentation package.
- Updated dependencies [5a075b9]
  - @arizeai/openinference-semantic-conventions@2.12.0
  - @arizeai/openinference-core@2.7.1
