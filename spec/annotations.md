# Annotations and Evaluations

Generative applications are judged by people, language models, and deterministic code. The resulting feedback is part
of the observability record: it must be possible to export a judgment with a span, import it into another system, and
preserve the meaning of its score, label, explanation, source, and metadata.

OpenInference represents this feedback in two contexts:

- An **annotation** is feedback attached to the span being judged.
- An **evaluation** is feedback produced by an `EVALUATOR` span.

Both contexts use the same fields. They differ only in which span carries the fields and what the fields describe.

## Attribute Patterns

Annotations use the following flattened attribute pattern:

`annotations.<index>.annotation.<attribute>`

Evaluations use the symmetric pattern:

`evaluations.<index>.evaluation.<attribute>`

`<index>` MUST be a zero-based integer. Producers SHOULD use contiguous indices beginning with `0`. The ordering of
annotation and evaluation objects has no semantic meaning.

### Feedback Attributes

| Attribute | Type | Requirement | Description |
| --- | --- | --- | --- |
| `name` | String | Required | Name of the criterion or metric being judged, such as `hallucination`, `correctness`, or `user_satisfaction`. |
| `score` | Integer or Float | Conditionally required | Numeric result of the judgment. Required when both `label` and `explanation` are absent. |
| `label` | String | Conditionally required | Categorical result of the judgment. Required when both `score` and `explanation` are absent. |
| `explanation` | String | Conditionally required | Human-readable explanation or evidence for the result. Required when both `score` and `label` are absent. |
| `annotator_kind` | String | Recommended | Kind of judge that produced the result. See [Annotator Kinds](#annotator-kinds). |
| `identifier` | String | Optional | Producer-assigned stable identifier for distinguishing or updating results with the same `name` on the same target. |
| `metadata` | JSON String | Optional | Valid JSON object containing additional result or annotator information not represented by a first-class attribute. |

Each object MUST contain `name` and at least one of `score`, `label`, or `explanation`. An object MAY contain any
combination of these result fields. A producer MUST omit fields for which it has no value; it MUST NOT encode an absent
value as an empty string or JSON `null`.

The interpretation and range of `score` are defined by `name` and the evaluator. Scores are not assumed to be
normalized, and a greater value is not assumed to be better. When a score is not self-describing, producers SHOULD
include a low-cardinality `label` and SHOULD document the score range and direction. `metadata` MAY carry information
such as the evaluator name and version, rubric version, score bounds, or threshold.

Labels SHOULD have low cardinality and stable spelling for a given annotation name. Explanations are intended for
display and investigation and MUST NOT be used as categorical values.

### Annotator Kinds

`annotator_kind` has the following case-sensitive, well-known values. If one applies, the respective value MUST be used;
otherwise, a custom value MAY be used.

| Value | Description |
| --- | --- |
| `HUMAN` | A person supplied the judgment, including explicit user feedback or expert review. |
| `LLM` | A language model supplied the judgment, including an LLM-as-judge. |
| `CODE` | Deterministic or statistical code supplied the judgment without an LLM or direct human decision. |

## Annotations on the Judged Span

When feedback is available before a span ends, it SHOULD be recorded on the span being judged using `annotations.*`.
Annotations MAY be attached to any OpenInference span kind.

```json
{
  "name": "answer_question",
  "attributes": {
    "openinference.span.kind": "CHAIN",
    "annotations.0.annotation.name": "hallucination",
    "annotations.0.annotation.score": 1,
    "annotations.0.annotation.label": "hallucinated",
    "annotations.0.annotation.explanation": "The answer contains a claim not supported by the retrieved documents.",
    "annotations.0.annotation.annotator_kind": "LLM",
    "annotations.0.annotation.identifier": "judge-v2",
    "annotations.0.annotation.metadata": "{\"evaluator\":\"judge-v2\",\"rubric_version\":\"2026-07-01\"}"
  }
}
```

An annotation describes the span that carries it. It does not describe the process that computed the result. When the
evaluation process itself must be traced, producers SHOULD emit an `EVALUATOR` span as described below and SHOULD use
that span as the canonical transport for the result to avoid duplication.

## Evaluations Produced by an Evaluator Span

An operation that computes feedback SHOULD be represented by a span with
`openinference.span.kind = "EVALUATOR"`. Results produced by that operation SHOULD be recorded using `evaluations.*`.
An evaluator span MAY produce more than one evaluation result about the same target span.

```json
{
  "name": "hallucination_judge",
  "context": {
    "trace_id": "5b8aa5a2d2c872e8321cf37308d69df2",
    "span_id": "93564f51e1abe1c2"
  },
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
    "evaluations.0.evaluation.explanation": "The answer contains a claim not supported by the retrieved documents.",
    "evaluations.0.evaluation.annotator_kind": "LLM",
    "evaluations.0.evaluation.identifier": "judge-v2",
    "evaluations.0.evaluation.metadata": "{\"evaluator\":\"judge-v2\",\"rubric_version\":\"2026-07-01\"}"
  }
}
```

The evaluator span SHOULD contain exactly one OpenTelemetry Span Link identifying the span that all of its
`evaluations.*` results evaluate. A parent-child relationship MAY additionally be used when it truthfully represents
the execution relationship, but parentage alone MUST NOT be used to infer which span was evaluated. When one evaluator
operation judges multiple target spans, the producer SHOULD emit one `EVALUATOR` span per target so that each result is
unambiguous. This is especially important when evaluation occurs after the original trace has completed.

## Transport and Lifecycle

OpenTelemetry spans are immutable after they end. Feedback obtained after the target span has ended MUST NOT be added by
mutating or re-exporting the original span. The feedback SHOULD instead be transported on a new `EVALUATOR` span using
`evaluations.*`, with a Span Link to the target span.

This supports both common transport paths:

1. **Inline feedback:** a producer attaches `annotations.*` before the target span ends.
2. **Out-of-band feedback:** a producer emits an `EVALUATOR` span containing `evaluations.*` and links it to the target.

A receiving system MAY materialize a linked evaluation as an annotation on its stored representation of the target
span. When exporting that feedback through OTLP after the target span has ended, the system SHOULD emit an `EVALUATOR`
span and link it to the target. It MUST preserve the feedback field values and the association with the target.

Producers SHOULD select one transport path for each result. If the same result is emitted as both an annotation and an
evaluation, corresponding fields MUST have identical values. When `identifier` is present, consumers MAY use the tuple
of target span, `name`, and `identifier` as the result identity for idempotent updates or deduplication. When it is
absent, no portable result identity is defined. List indices MUST NOT be treated as stable identifiers.

Because OTLP span attributes support only primitive values and homogeneous arrays, each feedback object MUST be
flattened into individual attributes. `metadata` MUST be a valid JSON object serialized as a string; language-specific
map representations such as `{'foo': 'bar'}` are not valid JSON. Exporters and collectors MUST preserve numeric values
as numeric OTLP attribute values rather than converting them to strings.

## Privacy and Limits

Explanations and metadata can contain application content, personal information, or evaluator reasoning. Producers MUST
apply the same masking, redaction, and access-control policies used for other sensitive span content.

Each feedback object expands to multiple span attributes and is subject to OpenTelemetry attribute-count and value-size
limits. Producers SHOULD budget for the complete object. If limits prevent emitting a valid object containing `name`
and at least one of `score`, `label`, or `explanation`, the producer SHOULD omit that object rather than emit an
incomplete result.

## OpenTelemetry GenAI Evaluation Compatibility

OpenTelemetry's GenAI semantic conventions define a `gen_ai.evaluation.result` event with the following equivalent
fields. Gateways MAY use this mapping when translating between OpenInference span payloads and OpenTelemetry GenAI
events.

| OpenInference feedback field | OpenTelemetry GenAI event attribute |
| --- | --- |
| `name` | `gen_ai.evaluation.name` |
| `score` | `gen_ai.evaluation.score.value` |
| `label` | `gen_ai.evaluation.score.label` |
| `explanation` | `gen_ai.evaluation.explanation` |

`annotator_kind`, `identifier`, and `metadata` do not have direct equivalents in that event schema and SHOULD be
preserved when the destination permits additional attributes.
