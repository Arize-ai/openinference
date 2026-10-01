# Decision Spans

Decision spans capture calls to a decision model: a model that scores or selects among candidate options supplied in the request rather than generating free-form text. Examples include choosing a route, judging whether a condition holds, or scoring an item against a rubric. Decision models (for example the TypeSafe `jev` family) are not language models, so decision spans use a dedicated `decision.*` namespace for model identification instead of `llm.*`.

## Required Attributes

All decision spans MUST include:

- `openinference.span.kind`: Set to `"DECISION"`
- `decision.system`: The AI system/product serving the decision model (e.g., "typesafe")

## Common Attributes

Decision spans typically include:

- `decision.model_name`: The decision model used (e.g., "jev-0123")
- `decision.request.model_name`: The model requested by the caller, when it can differ from the model that served the response (e.g., an alias such as "jev-latest")
- `decision.response.model_name`: The model that actually produced the decision, as reported by the provider (e.g., "jev-0123")
- `decision.provider`: The hosting provider of the decision model, when different from the system (e.g., "typesafe")
- `input.value`: The raw request as a JSON string, including the candidate options
- `input.mime_type`: Usually "application/json"
- `output.value`: The raw response as a JSON string, including the selection or scores
- `output.mime_type`: Usually "application/json"

## Model Identification

The `decision.*` identification attributes mirror their `llm.*` counterparts and follow the same rules:

| Decision attribute             | LLM counterpart           |
| ------------------------------ | ------------------------- |
| `decision.system`              | `llm.system`              |
| `decision.provider`            | `llm.provider`            |
| `decision.model_name`          | `llm.model_name`          |
| `decision.request.model_name`  | `llm.request.model_name`  |
| `decision.response.model_name` | `llm.response.model_name` |

- `decision.system` and `decision.provider` MUST use the well-known values listed for `llm.system` and `llm.provider` in the [Semantic Conventions](./semantic_conventions.md#reserved-attributes) when one applies; otherwise a custom value MAY be used.
- `decision.request.model_name` and `decision.response.model_name` are optional. Set them only when the response distinguishes the requested model from the model that served it, for example when the caller requests a floating alias (`jev-latest`) and the provider reports the pinned version that answered (`jev-0123`).
- `decision.model_name` SHOULD equal `decision.response.model_name` when known, falling back to `decision.request.model_name` otherwise, so consumers that only read `decision.model_name` see the most specific model identifier available.

## Attributes Not Used in Decision Spans

Decision spans SHOULD NOT set `llm.system`, `llm.provider`, `llm.model_name`, `llm.request.model_name`, or `llm.response.model_name`. Those attributes identify language models; using them on decision spans conflates decision model usage with LLM usage in downstream analytics such as model-level cost and token reporting.

### Transition Note

This section is not yet normative. Instrumentations written before the `DECISION` span kind and the `decision.*` attributes existed (for example, the OpenInference TypeSafe instrumentors) currently record decision model calls as `LLM` spans carrying `llm.system`, `llm.provider`, and `llm.model_name`. Those instrumentations are expected to migrate to `DECISION` spans with `decision.*` attributes in a follow-up release. Until that migration ships, consumers SHOULD accept both representations, and the `SHOULD NOT` above is guidance for new instrumentations rather than a conformance requirement for existing ones.

## Context Attributes

Decision spans inherit the same context attributes as every other OpenInference span (`session.id`, `user.id`, `metadata`, `tag.tags`) when they are set via the instrumentation context API. See [Configuration](./configuration.md) for details.

## Example

```
openinference.span.kind = "DECISION"
decision.system = "typesafe"
decision.provider = "typesafe"
decision.request.model_name = "jev-latest"
decision.response.model_name = "jev-0123"
decision.model_name = "jev-0123"
input.mime_type = "application/json"
input.value = "{\"condition\": \"The response cites a source\", \"candidates\": [\"yes\", \"no\"]}"
output.mime_type = "application/json"
output.value = "{\"decision\": \"yes\", \"score\": 0.93}"
```
