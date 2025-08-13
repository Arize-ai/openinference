# @arizeai/openinference-semantic-conventions

## 2.1.1

### Patch Changes

- 59be946: Initial Bedrock Invoke and Converse JS Instrumentation

## 2.1.0

### Minor Changes

- 34a4159: feat: add provider enums for xai and deepseek

## 2.0.0

### Major Changes

- c2ee804: Update the input and output to be under details
- 5f904bf: Update semantic conventions to include detailed token count and cost attributes, adding support for tracking token usage details, cache performance, and cost breakdowns in LLM operations.

### Minor Changes

- 5f90a80: Adding agent and graph semantic conventions

## 1.1.0

### Minor Changes

- ae5cd15: add semantic conventions for audio token count

## 1.0.1

### Patch Changes

- c4e2252: add semantic conventions to capture details in llm token counts: cached and reasoning

## 1.0.0

### Major Changes

- 16a3815: ESM support

  Packages are now shipped as "Dual Package" meaning that ESM and CJS module resolution
  should be supported for each package.

  Support is described as "experimental" because opentelemetry describes support for autoinstrumenting
  ESM projects as "ongoing". See https://github.com/open-telemetry/opentelemetry-js/blob/61d5a0e291db26c2af638274947081b29db3f0ca/doc/esm-support.md

## 0.14.0

### Minor Changes

- 1188c6d: add semantic conventions for audio

## 0.13.0

### Minor Changes

- 710d1d3: Add llm.system and llm.provider to LLMAttributePostfixes record

## 0.12.0

### Minor Changes

- a0e6f30: Support tool_call_id and tool_call.id

## 0.11.0

### Minor Changes

- f965410: Add system and provider attributes to openai spans
- d200d85: Add semantic conventions for llm.system and llm.provider

## 0.10.0

### Minor Changes

- ba142d5: Added attributes for tools and their json_schema

## 0.9.0

### Minor Changes

- 28a4ea2: adds missing evaluator span kind, reranker, metadata, tag, and tool parameter semantic conventions
- 96af3d6: export image conventions

## 0.8.0

### Minor Changes

- 9affdf6: fix: correct spelling of image attributes

## 0.7.0

### Minor Changes

- 9c44b14: Release guardrail span kinds

## 0.6.0

### Minor Changes

- b66bf54: Add Guardrail span kind
- 60ade67: Add multimodal semantic conventions for llm message contents

### Patch Changes

- fe69250: publish guardrail spankind

## 0.5.0

### Minor Changes

- 0d1d065: Add resource attribute for project name

## 0.4.0

### Minor Changes

- 921a40c: Add semantic conventions for session.id and user.id

## 0.3.0

### Minor Changes

- 9af89e5: add semantic conventions for prompt template, retrieval documents, tools, and functions

## 0.2.0

### Minor Changes

- 04c303f: Add metadata and tag semantic conventions

## 0.1.0

### Minor Changes

- 1925aad: add llm.prompts semantic convention

## 0.0.13

### Patch Changes

- 9b3bc4a: Add OpenAI Embeddings sementic attributes and instrumentation

## 0.0.12

### Patch Changes

- Add embededing vector semantic convention

## 0.0.11

### Patch Changes

- Token count support for OpenAI
- OpenAI instrumentation
