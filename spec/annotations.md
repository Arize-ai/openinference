# Annotations and Evaluations

Human, LLM, and code judges produce feedback about generative applications. Some systems call this feedback
annotations; others call the same grading results evaluations or evals. OpenInference supports both terms:

| Prefix | Meaning |
| --- | --- |
| `annotations.<index>.annotation` | Feedback using annotation terminology. |
| `evaluations.<index>.evaluation` | Feedback using evaluation or eval terminology. |

Both forms use the fields below. `<index>` MUST be zero-based and SHOULD be contiguous. Order has no meaning.

## Fields

| Field | Type | Requirement | Meaning |
| --- | --- | --- | --- |
| `name` | String | Required | Criterion or metric, such as `hallucination` or `correctness`. |
| `score` | Integer or Float | Optional | Numeric result. |
| `label` | String | Optional | Categorical result. |
| `explanation` | String | Optional | Human-readable reason or evidence. |
| `annotator_kind` | String | Recommended | Kind of judge. See [Annotator Kinds](#annotator-kinds). |
| `identifier` | String | Optional | Stable producer-assigned ID for results with the same `name` and target. |
| `metadata` | JSON String | Optional | Additional result or annotator data as a valid JSON object. |

Each object MUST have `name` and at least one of `score`, `label`, or `explanation`. Missing fields MUST be omitted.

The meaning, range, and direction of `score` depend on `name` and the evaluator; scores are not assumed to be normalized
or ordered from worse to better. Labels SHOULD have low cardinality and stable spelling.

### Annotator Kinds

`annotator_kind` MUST use these case-sensitive values when applicable; otherwise, a custom value MAY be used.

| Value | Meaning |
| --- | --- |
| `HUMAN` | A person judged the span. |
| `LLM` | A language model judged the span. |
| `CODE` | Deterministic or statistical code judged the span. |

## Transport

Attach results to the span they describe. Use the form that matches the producing system's terminology; consumers
SHOULD treat both forms as equivalent.

Annotation example:

```json
{
  "attributes": {
    "openinference.span.kind": "CHAIN",
    "annotations.0.annotation.name": "hallucination",
    "annotations.0.annotation.score": 1,
    "annotations.0.annotation.label": "hallucinated",
    "annotations.0.annotation.explanation": "The claim is not supported by the retrieved documents.",
    "annotations.0.annotation.annotator_kind": "LLM",
    "annotations.0.annotation.identifier": "judge-v2",
    "annotations.0.annotation.metadata": "{\"rubric_version\":\"2\"}"
  }
}
```

The same feedback may use evaluation terminology:

```json
{
  "attributes": {
    "evaluations.0.evaluation.name": "hallucination",
    "evaluations.0.evaluation.score": 1,
    "evaluations.0.evaluation.label": "hallucinated",
    "evaluations.0.evaluation.annotator_kind": "LLM"
  }
}
```

Producers SHOULD use one form per result. If both forms carry the same result, their fields MUST match. When
`identifier` is present, consumers MAY identify the result by target span, `name`, and `identifier`. List indices are
not identifiers.

## Encoding and Privacy

Feedback objects MUST be flattened because OTLP span attributes cannot contain objects. `metadata` MUST be a JSON object
serialized as a string, and scores MUST remain numeric OTLP values.

Explanations and metadata can contain sensitive data and MUST follow the same masking and access policies as other span
content. If attribute limits prevent a complete object, producers SHOULD omit it rather than emit an object without
`name` or a result field.

## Single-Result Events

A system using the OpenTelemetry GenAI conventions may carry one result as a
[`gen_ai.evaluation.result`](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-events.md#event-gen_aievaluationresult)
event. For example, a span payload may contain:

```json
{
  "events": [
    {
      "name": "gen_ai.evaluation.result",
      "attributes": {
        "gen_ai.evaluation.name": "hallucination",
        "gen_ai.evaluation.score.value": 1,
        "gen_ai.evaluation.score.label": "hallucinated",
        "gen_ai.evaluation.explanation": "The claim is not supported by the retrieved documents."
      }
    }
  ]
}
```

Gateways MAY translate between this event and either OpenInference form:

| OpenInference field | OpenTelemetry attribute |
| --- | --- |
| `name` | `gen_ai.evaluation.name` |
| `score` | `gen_ai.evaluation.score.value` |
| `label` | `gen_ai.evaluation.score.label` |
| `explanation` | `gen_ai.evaluation.explanation` |

The OpenTelemetry event has no direct equivalent for `annotator_kind`, `identifier`, or `metadata`; gateways SHOULD
preserve them when the destination allows additional attributes.
