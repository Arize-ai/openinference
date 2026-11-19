# Annotations and Evaluations

Human, LLM, and code judges produce feedback about generative applications. Some systems call this feedback
annotations; others call the same grading results evaluations or evals. OpenInference supports both terms:

| Scope | Annotation form | Evaluation form | Target |
| --- | --- | --- | --- |
| Span | `annotations.<index>.annotation` | `evaluations.<index>.evaluation` | One span. |
| Trace | `trace.annotations.<index>.annotation` | `trace.evaluations.<index>.evaluation` | One trace. |
| Session | `session.annotations.<index>.annotation` | `session.evaluations.<index>.evaluation` | One session identified by `session.id`. |

All forms use the fields below. Within each collection, `<index>` MUST be zero-based and SHOULD be contiguous. Order has
no meaning.

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
| `HUMAN` | A person judged the target. |
| `LLM` | A language model judged the target. |
| `CODE` | Deterministic or statistical code judged the target. |

## Scope

Producers SHOULD use the narrowest applicable scope and SHOULD record each result once per target:

- Span feedback describes one span.
- Trace feedback describes the carrying span's trace when inline and the linked span's trace when post-hoc.
- Session feedback describes the session identified by `session.id`; the carrying span MUST include `session.id`.

Consumers MUST NOT apply trace or session feedback separately to each span as if it were duplicated span feedback.

## Transport

Use the form that matches the producing system's terminology; consumers SHOULD treat both forms as equivalent.

### Inline Feedback

When feedback is available while a suitable span is open:

- Record span feedback on the target span.
- Record trace feedback once on a span in the target trace, preferably the root span.
- Record session feedback once on a span carrying the target's `session.id`.

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

Trace and session feedback add a scope prefix:

```json
{
  "attributes": {
    "session.id": "session-123",
    "trace.annotations.0.annotation.name": "retrieval_quality",
    "trace.annotations.0.annotation.score": 0.92,
    "trace.annotations.0.annotation.annotator_kind": "CODE",
    "session.annotations.0.annotation.name": "conversational_coherence",
    "session.annotations.0.annotation.label": "coherent",
    "session.annotations.0.annotation.annotator_kind": "HUMAN"
  }
}
```

Producers SHOULD use one form per result. If both forms carry the same result, their fields MUST match. When
`identifier` is present, consumers MAY identify the result by scope, target, `name`, and `identifier`. List indices are
not identifiers.

### Post-Hoc Feedback

Ended spans are immutable. Post-hoc feedback MUST use a new carrier span. Target correlation depends on scope:

- A span-scoped carrier MUST have exactly one OpenTelemetry Span Link to the target span.
- A trace-scoped carrier MUST have exactly one Span Link to a span in the target trace and SHOULD link its root span.
  The link's trace ID identifies the target.
- A session-scoped carrier MUST include the target's `session.id`. It MAY link a representative span, but the link does
  not identify the session because one session can contain multiple traces.

A carrier SHOULD contain feedback for one scope and target. Its feedback describes the correlated target, not the
carrier. Producers SHOULD emit one carrier per target, and exporters MUST preserve the required link or `session.id`.

The carrier SHOULD use `openinference.span.kind = "EVALUATOR"` when it traces evaluation work. This span kind does not
determine whether the feedback uses annotation or evaluation terminology. Parentage MAY describe how the carrier was
created but MUST NOT be used to identify the feedback target.

```json
{
  "name": "hallucination evaluation",
  "links": [
    {
      "trace_id": "80f198ee56343ba864fe8b2a57d3eff7",
      "span_id": "e457b5a2e4d86bd1"
    }
  ],
  "attributes": {
    "openinference.span.kind": "EVALUATOR",
    "evaluations.0.evaluation.name": "hallucination",
    "evaluations.0.evaluation.label": "hallucinated",
    "evaluations.0.evaluation.annotator_kind": "HUMAN"
  }
}
```

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

If the target span has ended, this event MUST be recorded on the linked carrier described above. The GenAI event does
not encode trace or session scope, so gateways MUST preserve scoped OpenInference attributes when that scope cannot be
represented by the destination.

Gateways MAY translate between this event and either OpenInference form:

| OpenInference field | OpenTelemetry attribute |
| --- | --- |
| `name` | `gen_ai.evaluation.name` |
| `score` | `gen_ai.evaluation.score.value` |
| `label` | `gen_ai.evaluation.score.label` |
| `explanation` | `gen_ai.evaluation.explanation` |

The OpenTelemetry event has no direct equivalent for `annotator_kind`, `identifier`, or `metadata`; gateways SHOULD
preserve them when the destination allows additional attributes.
