# @arizeai/openinference-instrumentation-anthropic

## 1.0.0

### Major Changes

- 38b858d: Support the current Anthropic Messages API (stable and beta, `@anthropic-ai/sdk` 0.133)
  
  Behavior changes to note:
  
  - A streamed response's `output.value` is now the message rebuilt from the stream as JSON (`output.mime_type` `application/json`, previously the concatenated text as `text/plain`), the same as for a non-streaming response.
  - The top-level `system` prompt is now recorded as input message 0, so the conversation's messages start at index 1 when a system prompt is set. It is no longer part of `llm.invocation_parameters`, which `hideInputs` does not hide.
  
  Changes:
  
  - Streams are observed through the caller's own iterator instead of a tee, so a mid-stream error no longer causes an unhandled rejection, `break` still aborts the request, nothing is buffered, and the span ends OK, unset (left early or aborted) or ERROR. Streamed tool input is parsed once per block.
  - Attribute extraction can no longer throw into the caller.
  - Server and MCP tool calls are recorded as tool calls; compaction blocks as text; every other content block (tool results, documents, MCP listings, container uploads, unknown types) as its type and JSON, with embedded base64 data replaced by a placeholder and nested images hidden by `hideInputImages`.
  - Citations, MCP server names, tool call callers and toolset names, `stop_details`, cache miss reasons, the container id, context management edits, input transformations, iteration usage and server tool usage go to span metadata, merged with the context metadata. Citations are left out when inputs, outputs, input messages or output messages are hidden, and a citation's `cited_text` when input text or output text are hidden, and a refusal's `explanation` when outputs, output messages or output text are hidden.
  - Images record URLs and base64 data URIs (Files API images are recorded as their JSON); `llm.tools` records `tool.name` and `tool.description`, and applies the request's `tool_addition` and `tool_removal` blocks.
  - Thinking tokens are recorded as `llm.token_count.completion_details.reasoning`.
  - `llm.provider` is `aws`, `google` or `azure` for the Bedrock, Vertex and Foundry clients.
  - `messages.countTokens` / `beta.messages.countTokens` are traced, and each run of `beta.messages.toolRunner` runs under a `beta.messages.tool_runner` CHAIN span (the naming proposed for the Python instrumentation in PR #3531) parenting its LLM spans and a TOOL span per tool run, including tools added with `addTools()` or `setMessagesParams()`.

### Patch Changes

- 2925fc3: Record every `tool_result` block of an input message as its own tool message. Previously only the last block survived on the span, so parallel tool-call replies lost all but one result. Tool messages are recorded ahead of any other content in the same message, matching the order the API requires.
- Updated dependencies [dc63d01]
- Updated dependencies [298ff79]
  - @arizeai/openinference-semantic-conventions@2.15.0
  - @arizeai/openinference-core@2.8.1

## 0.2.11

### Patch Changes

- Updated dependencies [0d26a59]
  - @arizeai/openinference-core@2.8.0

## 0.2.10

### Patch Changes

- Updated dependencies [53b7a0e]
  - @arizeai/openinference-semantic-conventions@2.14.0
  - @arizeai/openinference-core@2.7.3

## 0.2.9

### Patch Changes

- Updated dependencies [a1f276c]
- Updated dependencies [a719562]
  - @arizeai/openinference-semantic-conventions@2.13.0
  - @arizeai/openinference-core@2.7.2

## 0.2.8

### Patch Changes

- Updated dependencies [5a075b9]
  - @arizeai/openinference-semantic-conventions@2.12.0
  - @arizeai/openinference-core@2.7.1

## 0.2.7

### Patch Changes

- 4157dde: Capture prompt caching token counts as `llm.token_count.prompt_details.cache_write` and `llm.token_count.prompt_details.cache_read` for streaming and non-streaming `messages.create` calls. Anthropic's `input_tokens` excludes cached tokens, so `llm.token_count.prompt` and `llm.token_count.total` now include the cache write and read counts. Zero cache counts are omitted, and streaming usage is merged field by field across `message_start`, the server-side fallback hop and `message_delta`, so streaming and non-streaming spans report the same counts.

## 0.2.6

### Patch Changes

- Updated dependencies [0ff0af2]
- Updated dependencies [0ff0af2]
  - @arizeai/openinference-semantic-conventions@2.11.0
  - @arizeai/openinference-core@2.7.0

## 0.2.5

### Patch Changes

- Updated dependencies [6d9f813]
  - @arizeai/openinference-semantic-conventions@2.10.0
  - @arizeai/openinference-core@2.6.3

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
