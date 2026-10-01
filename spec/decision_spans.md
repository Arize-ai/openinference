# Decision Spans

Decision spans capture calls to a decision model: a model that scores or selects among candidate options supplied in the request rather than generating free-form text. Examples include choosing a route, judging whether a condition holds, or scoring an item against a rubric. Decision models are not language models, so decision spans use a dedicated `decision.*` namespace for model identification instead of `llm.*`.

## Background

A decision model takes unstructured or structured state plus a set of typed questions and returns a typed, probabilistic answer for each question. The set of possible answers is fixed in advance by the caller, so the model cannot produce malformed output, and every answer carries a calibrated probability or confidence score that software can act on directly. Decision models trade free-form generation for speed, cost, and predictability, which makes them a fit for "smart if-statements": routing, classification, guardrails, scoring, extraction, and other branching decisions inside an application.

Representative decision model families:

- **TypeSafe AI System One models (Jev).** TypeSafe describes System One models as "a new class of frontier models built to make fast, structured decisions that software can use directly": unstructured state in, typed probabilistic decisions out, with no string generation. [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) is their first public model and is served through the [`system_one` API](https://docs.typesafe.ai/). A System One request sends a `state` plus a map of typed questions of three kinds (Noul for yes/no conditions, Choice for picking among options, Score for ordered rubrics) and returns one typed answer per question. Callers typically request the floating alias `jev-latest`, and the response reports the resolved version that answered, for example `jev-1.13.0`. The model's pricing and provider coverage are catalogued at [models.dev](https://models.dev/models/typesafe/jev-latest/).
- **OpenAI Decisions API.** Announced at [DevDay 2026](https://openai.com/index/devday-2026-recap/), the Decisions API "enables real-time decision-making by focusing Luna's intelligence on a specific set of user-defined questions with finite pre-defined answers." Developers supply context as text or images and get back answers they can use to classify content, route requests, or choose an agent's next action. It launched in limited preview; the public announcement does not yet document an endpoint path, model identifier, or response schema, so `decision.system` / `decision.provider` for these spans should use the well-known value `openai` and the model attributes whatever identifier the API returns.
- **vLLM structured decisions.** vLLM serves Jev-style decisions over open models. Its [structured reads example](https://docs.vllm.ai/en/latest/examples/features/structured_diffusion/) exposes a Jev-compatible `POST /v1/systemone` endpoint that takes `model`, `state`, and `questions` and returns calibrated probabilities from a single forward pass, and [vllm-project/vllm#59365](https://github.com/vllm-project/vllm/issues/59365) proposes a first-class `/v1/decisions` route with `/v1/systemone` kept as a compatibility layer. Spans for these calls use the self-hosted model's name and whichever `decision.provider` applies to the deployment.
- **vLLM Semantic Router Decision 1.0.** An open-weight family of six "Decision Foundation Models" (Kai, Lex, Eos, Sol, Nox, Lux) that exposes the same shape of interface: "state + questions + criteria → typed probability distributions", with Choice, Noul, and Score answer types. See the [Decision 1.0 announcement](https://vllm-sr.ai/blog/decision-models/).

OpenInference models these calls as `DECISION` spans rather than `LLM` spans because the semantics differ in ways that matter to observability tooling: there are no input or output messages, output tokens are often absent or free, model names and pricing live in a separate catalogue from language models, and the interesting output is a distribution over caller-supplied options rather than generated text. The design discussion is in [Arize-ai/openinference#3808](https://github.com/Arize-ai/openinference/issues/3808) and [Arize-ai/openinference#3894](https://github.com/Arize-ai/openinference/issues/3894).

## Required Attributes

All decision spans MUST include:

- `openinference.span.kind`: Set to `"DECISION"`
- `decision.system`: The AI system/product serving the decision model (e.g., "typesafe")

## Common Attributes

Decision spans typically include:

- `decision.model_name`: The decision model used (e.g., "jev-1.13.0")
- `decision.request.model_name`: The model requested by the caller, when it can differ from the model that served the response (e.g., an alias such as "jev-latest")
- `decision.response.model_name`: The model that actually produced the decision, as reported by the provider (e.g., "jev-1.13.0")
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
- `decision.request.model_name` and `decision.response.model_name` are optional. Set them only when the response distinguishes the requested model from the model that served it, for example when the caller requests a floating alias (`jev-latest`) and the provider reports the pinned version that answered (`jev-1.13.0`).
- `decision.model_name` SHOULD equal `decision.response.model_name` when known, falling back to `decision.request.model_name` otherwise, so consumers that only read `decision.model_name` see the most specific model identifier available.

## Attributes Not Used in Decision Spans

Decision spans SHOULD NOT set `llm.system`, `llm.provider`, `llm.model_name`, `llm.request.model_name`, or `llm.response.model_name`. Those attributes identify language models; using them on decision spans conflates decision model usage with LLM usage in downstream analytics such as model-level cost and token reporting.

### Transition Note

This section is not yet normative. Instrumentations written before the `DECISION` span kind and the `decision.*` attributes existed (for example, the OpenInference TypeSafe instrumentors for [Python](../python/instrumentation/openinference-instrumentation-typesafe) and [JavaScript](../js/packages/openinference-instrumentation-typesafe)) currently record decision model calls as `LLM` spans carrying `llm.system`, `llm.provider`, and `llm.model_name`. Those instrumentations are expected to migrate to `DECISION` spans with `decision.*` attributes in a follow-up release. Until that migration ships, consumers SHOULD accept both representations, and the `SHOULD NOT` above is guidance for new instrumentations rather than a conformance requirement for existing ones.

## Context Attributes

Decision spans inherit the same context attributes as every other OpenInference span (`session.id`, `user.id`, `metadata`, `tag.tags`) when they are set via the instrumentation context API. See [Configuration](./configuration.md) for details.

## Example

A TypeSafe System One call that asks Jev one Noul (yes/no) question about a piece of state. The caller requested the `jev-latest` alias and the provider answered with `jev-1.13.0`, so both model attributes are set and `decision.model_name` carries the resolved version.

```
openinference.span.kind = "DECISION"
decision.system = "typesafe"
decision.provider = "typesafe"
decision.request.model_name = "jev-latest"
decision.response.model_name = "jev-1.13.0"
decision.model_name = "jev-1.13.0"
input.mime_type = "application/json"
input.value = "{\"model\": \"jev-latest\", \"state\": \"The assistant replied: 'Per the 2024 audit (p. 12), revenue grew 8%.'\", \"questions\": {\"cites_source\": {\"type\": \"noul\", \"question\": \"Does the response cite a source?\"}}}"
output.mime_type = "application/json"
output.value = "{\"model\": \"jev-1.13.0\", \"answers\": {\"cites_source\": {\"answer\": true, \"probability\": 0.93}}}"
```

## References

- TypeSafe AI, [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (September 2026): the announcement of the System One model class and the Jev model.
- TypeSafe AI, [documentation](https://docs.typesafe.ai/): the `system_one` API and the Noul, Choice, and Score question types.
- models.dev, [typesafe/jev-latest](https://models.dev/models/typesafe/jev-latest/): model specs, pricing, and the providers that serve Jev.
- OpenAI, [DevDay 2026 Recap](https://openai.com/index/devday-2026-recap/): the Decisions API announcement (limited preview).
- vLLM, [Structured reads on DiffusionGemma](https://docs.vllm.ai/en/latest/examples/features/structured_diffusion/): the example server behind the Jev-compatible `/v1/systemone` endpoint, with the launch recipe at [recipes.vllm.ai](https://recipes.vllm.ai/Google/diffusiongemma-26B-A4B-it#jev-style-structured-decisions-nightly).
- vLLM, [RFC: /v1/decisions](https://github.com/vllm-project/vllm/issues/59365): the proposal for a first-class typed decision endpoint in the vLLM API server, and [vllm-project/vllm#59299](https://github.com/vllm-project/vllm/pull/59299), the `/v1/systemone` structured decisions endpoint.
- vLLM Semantic Router, [Introducing Decision 1.0: Open Decision Foundation Models](https://vllm-sr.ai/blog/decision-models/): an open-weight decision model family with the same Choice, Noul, and Score interface.
- OpenInference, [Arize-ai/openinference#3808](https://github.com/Arize-ai/openinference/issues/3808) (decision model semantics) and [Arize-ai/openinference#3894](https://github.com/Arize-ai/openinference/issues/3894) (decision attributes): the discussions behind these conventions.
- OpenInference TypeSafe instrumentors for [Python](../python/instrumentation/openinference-instrumentation-typesafe) and [JavaScript](../js/packages/openinference-instrumentation-typesafe).
