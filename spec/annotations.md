# Annotations and Evaluations

Human, LLM, and code judges produce feedback about generative applications. OpenInference carries this feedback in two
equivalent forms:

| Prefix | Meaning |
| --- | --- |
| `annotations.<index>.annotation` | Feedback about the span that carries it. |
| `evaluations.<index>.evaluation` | Feedback produced by an `EVALUATOR` span. |

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

Use `annotations.*` when feedback is available before the target span ends:

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

Use `evaluations.*` on a span with `openinference.span.kind = "EVALUATOR"` when tracing the evaluation or when feedback
arrives after the target span ends:

```json
{
  "links": [
    {
      "trace_id": "80f198ee56343ba864fe8b2a57d3eff7",
      "span_id": "e457b5a2e4d86bd1"
    }
  ],
  "attributes": {
    "openinference.span.kind": "EVALUATOR",
    "evaluations.0.evaluation.name": "hallucination",
    "evaluations.0.evaluation.score": 1,
    "evaluations.0.evaluation.label": "hallucinated",
    "evaluations.0.evaluation.annotator_kind": "LLM"
  }
}
```

An evaluator span SHOULD have exactly one OpenTelemetry Span Link to its target. All `evaluations.*` on that span apply
to the linked target. Producers SHOULD emit one evaluator span per target. Parentage MAY describe execution but does not
identify the target.

Spans are immutable after they end, so post-hoc feedback MUST use a linked evaluator span rather than mutate or
re-export the target. Consumers MAY display a linked evaluation as an annotation, but exporters MUST preserve its fields
and target association.

Producers SHOULD use one form per result. If both forms carry the same result, their fields MUST match. When
`identifier` is present, consumers MAY identify the result by target span, `name`, and `identifier`. List indices are
not identifiers.

## Encoding and Privacy

Feedback objects MUST be flattened because OTLP span attributes cannot contain objects. `metadata` MUST be a JSON object
serialized as a string, and scores MUST remain numeric OTLP values.

Explanations and metadata can contain sensitive data and MUST follow the same masking and access policies as other span
content. If attribute limits prevent a complete object, producers SHOULD omit it rather than emit an object without
`name` or a result field.

## OpenTelemetry Compatibility

Gateways MAY translate the common fields to an OpenTelemetry `gen_ai.evaluation.result` event as follows:

| OpenInference field | OpenTelemetry attribute |
| --- | --- |
| `name` | `gen_ai.evaluation.name` |
| `score` | `gen_ai.evaluation.score.value` |
| `label` | `gen_ai.evaluation.score.label` |
| `explanation` | `gen_ai.evaluation.explanation` |

The OpenTelemetry event has no direct equivalent for `annotator_kind`, `identifier`, or `metadata`; gateways SHOULD
preserve them when the destination allows additional attributes.
