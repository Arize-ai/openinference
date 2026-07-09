# @arizeai/openinference-genai

## 0.3.0

### Minor Changes

- d0f5a88: feat(openinference-genai): Improve compatability with gen_ai conventions

## 0.2.0

### Minor Changes

- 52f368d: Map `gen_ai.operation.name` to OpenInference span kinds. Control-flow operations `"create_agent"`, `"invoke_agent"`, and `"plan"` map to `AGENT`, and `"invoke_workflow"` maps to `CHAIN`. Terminal (concrete leaf) operations are classified by operation name: `"chat"`, `"text_completion"`, and `"generate_content"` map to `LLM`; `"embeddings"` to `EMBEDDING`; `"retrieval"` to `RETRIEVER`; and `"execute_tool"` to `TOOL`.

  An explicit agent identity (a `gen_ai.agent.*` attribute) classifies a span as `AGENT`, but only for control-flow or unrecognized operations — on terminal operations the agent identity is treated as context and does not change the kind. `gen_ai.agent.name` is now mapped to `agent.name` via the new `mapAgentAttributes` helper.

  The minimum supported `@opentelemetry/semantic-conventions` peer dependency is now `1.41.1` (the first version exporting the `invoke_workflow` operation value). `"plan"` is matched by literal until a constant is published, since it was added in the OTel GenAI semantic conventions ([semantic-conventions-genai#97](https://github.com/open-telemetry/semantic-conventions-genai/pull/97)) but is not yet released.

  This reclassifies some spans that previously fell through to `LLM` (e.g. an `invoke_agent` span without agent attributes now becomes `AGENT`). `"plan"` is a value added in the OTel GenAI semantic conventions ([semantic-conventions-genai#97](https://github.com/open-telemetry/semantic-conventions-genai/pull/97)) that is not yet released, so it is matched by literal until a constant is published. The minimum supported `@opentelemetry/semantic-conventions` peer dependency is now `1.41.1` (the first version exporting the `invoke_workflow` operation value).

## 0.1.10

### Patch Changes

- Updated dependencies [0f0242c]
  - @arizeai/openinference-semantic-conventions@2.5.0

## 0.1.9

### Patch Changes

- Updated dependencies [81b8bdb]
  - @arizeai/openinference-semantic-conventions@2.4.0

## 0.1.8

### Patch Changes

- Updated dependencies [e09ce3f]
  - @arizeai/openinference-semantic-conventions@2.3.0

## 0.1.7

### Patch Changes

- Updated dependencies [7eb1c88]
  - @arizeai/openinference-semantic-conventions@2.2.0

## 0.1.6

### Patch Changes

- 7cd28bc: feat: Add commonjs builds to openinference-genai

## 0.1.5

### Patch Changes

- c79c564: force publish
- c79c564: signed publishing
- Updated dependencies [c79c564]
- Updated dependencies [c79c564]
  - @arizeai/openinference-semantic-conventions@2.1.7

## 0.1.4

### Patch Changes

- a4eead1: force publish
- a4eead1: signed publishing
- Updated dependencies [a4eead1]
- Updated dependencies [a4eead1]
  - @arizeai/openinference-semantic-conventions@2.1.6

## 0.1.3

### Patch Changes

- 74f278c: force publish
- 74f278c: signed publishing
- Updated dependencies [74f278c]
- Updated dependencies [74f278c]
  - @arizeai/openinference-semantic-conventions@2.1.5

## 0.1.2

### Patch Changes

- fe61379: force publish
- fe61379: signed publishing
- Updated dependencies [fe61379]
- Updated dependencies [fe61379]
  - @arizeai/openinference-semantic-conventions@2.1.4

## 0.1.1

### Patch Changes

- 006a685: signed publishing
- Updated dependencies [006a685]
  - @arizeai/openinference-semantic-conventions@2.1.3

## 0.1.0

### Minor Changes

- 59e5c8b: feat(openinference-genai): Create @arizeai/openinference-genai package
