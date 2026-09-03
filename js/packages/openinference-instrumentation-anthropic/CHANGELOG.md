# @arizeai/openinference-instrumentation-anthropic

## 0.2.4

### Patch Changes

- Updated dependencies [fd01216]
  - @arizeai/openinference-semantic-conventions@2.9.0
  - @arizeai/openinference-core@2.6.2

## 0.2.3

### Patch Changes

- Updated dependencies [4d72f42]
  - @arizeai/openinference-core@2.6.1

## 0.2.2

### Patch Changes

- Updated dependencies [99f6e71]
  - @arizeai/openinference-core@2.6.0

## 0.2.1

### Patch Changes

- 5f38a16: Scope the double-patch guard to the module object so both the CJS and ESM builds of a dual-package SDK can be patched in the same process. Previously the module-global `_isOpenInferencePatched` flag made whichever build was patched first silently block `patch()`/`manuallyInstrument()` for the other build (#3557). The guard is now a `WeakSet` keyed on the patched class, which needs no write to the module and therefore also keeps protecting immutable modules (Deno, webpack) — the case the global flag existed for. `isPatched()` behavior is unchanged.
- 0071b37: Split over-complex functions into focused helpers and make implicit returns explicit (enforce `eslint/complexity`). Also hardens bedrock-agent-runtime tool-call extraction against a `function: null` payload that previously threw. No other behavior changes.
- Updated dependencies [0071b37]
  - @arizeai/openinference-core@2.5.4

## 0.2.0

### Minor Changes

- a3c04ed: Instrument `beta.messages.create` and capture `llm.request.model_name` and `llm.response.model_name` for streaming and non-streaming stable and beta `messages.create` calls, alongside the existing `llm.model_name`. Server-side fallback is now traced end to end: the response model attributes are updated at fallback boundaries so they identify the model that actually served the response, the `fallback` content block is recorded as message content (with the declining model and refusal category) instead of leaving a hole in `message_contents`, token counts are taken from the serving attempt rather than mixed across models, and `llm.finish_reason` is recorded so classifier refusals (`stop_reason: "refusal"`) are visible on the span.

## 0.1.22

### Patch Changes

- Updated dependencies [1fe497f]
  - @arizeai/openinference-semantic-conventions@2.8.0
  - @arizeai/openinference-core@2.5.3

## 0.1.21

### Patch Changes

- 74ae809: Replace unsafe type assertions with runtime type guards across packages (enforce `typescript/no-unsafe-type-assertion`)
- Updated dependencies [74ae809]
  - @arizeai/openinference-core@2.5.2

## 0.1.20

### Patch Changes

- Updated dependencies [237ce2b]
  - @arizeai/openinference-semantic-conventions@2.7.0
  - @arizeai/openinference-core@2.5.1

## 0.1.19

### Patch Changes

- b067bbb: Preserve `APIPromise` helpers (`withResponse()` / `asResponse()`) on the patched `messages.create`, fixing `client.messages.stream()` throwing `create(...).withResponse is not a function` when instrumented.

## 0.1.18

### Patch Changes

- Updated dependencies [0168198]
  - @arizeai/openinference-core@2.5.0

## 0.1.17

### Patch Changes

- Updated dependencies [145e3c6]
  - @arizeai/openinference-semantic-conventions@2.6.0
  - @arizeai/openinference-core@2.4.1

## 0.1.16

### Patch Changes

- 15cddf4: Anthropic instrumentation now captures Claude extended thinking content in OpenInference message contents. Anthropic thinking blocks are recorded as reasoning content with their text and signature, while redacted_thinking blocks are recorded as reasoning content with their redacted data payload. This works for both streaming and non-streaming Messages responses, preserves content block ordering.

## 0.1.15

### Patch Changes

- Updated dependencies [d0f5a88]
  - @arizeai/openinference-core@2.4.0

## 0.1.14

### Patch Changes

- Updated dependencies [1fe7927]
  - @arizeai/openinference-core@2.3.0

## 0.1.13

### Patch Changes

- Updated dependencies [0f0242c]
- Updated dependencies [26733d8]
  - @arizeai/openinference-semantic-conventions@2.5.0
  - @arizeai/openinference-core@2.2.0

## 0.1.12

### Patch Changes

- Updated dependencies [81b8bdb]
  - @arizeai/openinference-semantic-conventions@2.4.0
  - @arizeai/openinference-core@2.1.1

## 0.1.11

### Patch Changes

- Updated dependencies [cfb128c]
  - @arizeai/openinference-core@2.1.0

## 0.1.10

### Patch Changes

- Updated dependencies [e09ce3f]
  - @arizeai/openinference-semantic-conventions@2.3.0
  - @arizeai/openinference-core@2.0.8

## 0.1.9

### Patch Changes

- Updated dependencies [4eebba3]
  - @arizeai/openinference-core@2.0.7

## 0.1.8

### Patch Changes

- Updated dependencies [7eb1c88]
- Updated dependencies [3944459]
  - @arizeai/openinference-semantic-conventions@2.2.0
  - @arizeai/openinference-core@2.0.6

## 0.1.7

### Patch Changes

- c79c564: force publish
- c79c564: signed publishing
- Updated dependencies [c79c564]
- Updated dependencies [c79c564]
  - @arizeai/openinference-core@2.0.5
  - @arizeai/openinference-semantic-conventions@2.1.7

## 0.1.6

### Patch Changes

- a4eead1: force publish
- a4eead1: signed publishing
- Updated dependencies [a4eead1]
- Updated dependencies [a4eead1]
  - @arizeai/openinference-core@2.0.4
  - @arizeai/openinference-semantic-conventions@2.1.6

## 0.1.5

### Patch Changes

- 74f278c: force publish
- 74f278c: signed publishing
- Updated dependencies [74f278c]
- Updated dependencies [74f278c]
  - @arizeai/openinference-core@2.0.3
  - @arizeai/openinference-semantic-conventions@2.1.5

## 0.1.4

### Patch Changes

- fe61379: force publish
- fe61379: signed publishing
- Updated dependencies [fe61379]
- Updated dependencies [fe61379]
  - @arizeai/openinference-core@2.0.2
  - @arizeai/openinference-semantic-conventions@2.1.4

## 0.1.3

### Patch Changes

- 006a685: signed publishing
- Updated dependencies [006a685]
  - @arizeai/openinference-core@2.0.1
  - @arizeai/openinference-semantic-conventions@2.1.3

## 0.1.2

### Patch Changes

- Updated dependencies [d3d7017]
  - @arizeai/openinference-core@2.0.0

## 0.1.1

### Patch Changes

- Updated dependencies [5161c9f]
  - @arizeai/openinference-core@1.0.8

## 0.1.0

### Minor Changes

- 43db6b3: - Support for `anthropic.messages.create()` method
  - Support for streaming responses
  - Tool use/function calling instrumentation
  - Token usage tracking
  - Full OpenInference semantic conventions compliance
  - TypeScript support
