# Classifier Spans — Proposal

`CLASSIFIER` captures binary judgments, categorical selections, and scoring against ordered
classes, regardless of provider. One call may produce several classifications over shared input.
Unlike [evaluation feedback](./annotations.md), these results describe the operation's input,
not the quality of a target span, trace, or session.

## Attributes

| Attribute | Type | Meaning |
| --- | --- | --- |
| `openinference.span.kind` | String | `CLASSIFIER` |
| `classifier.model_name` | String | Model used, when applicable |
| `classifier.invocation_parameters` | JSON string | Configuration excluding input content |
| `classifier.input.value` | String | Content being classified |
| `classifier.input.mime_type` | String | `text/plain` or `application/json` |

Retain `input.value` / `output.value` and their MIME types for raw request/response inspection.

Each task uses `classifier.classifications.{i}.classification.` followed by:

| Field | Type | Meaning |
| --- | --- | --- |
| `name` | String | Task identifier; required |
| `type` | String | `binary`, `categorical`, or `ordinal`; required |
| `instructions` | JSON string | Optional instructions; preserves strings, objects, or arrays |
| `probability` | Double | Binary probability of true, from 0 to 1 |
| `label` | String | Selected categorical class, when returned |
| `score` | Double | Ordinal result on the original numeric scale |
| `confidence` | Double | Optional producer-reported certainty from 0 to 1; not necessarily probability of correctness |
| `classes.{j}.class.label` | String | Class identifier; binary labels are `false` and `true` |
| `classes.{j}.class.description` | JSON string | Optional class criteria, including structured descriptions |
| `classes.{j}.class.value` | Double | Numeric position of an ordinal class |
| `classes.{j}.class.probability` | Double | Reported probability of that class |

Indices are zero-based. Match outputs to input task identifiers and class labels before
flattening; response order need not match request order. Omit unavailable fields.

## Example: one ticket, three judgments

Illustrative values, with optional configuration and raw payload attributes omitted:

```json
{
  "openinference.span.kind": "CLASSIFIER",
  "classifier.input.value": "Charged twice, and exports fail. I need them for a demo today.",
  "classifier.input.mime_type": "text/plain",

  "classifier.classifications.0.classification.name": "refund_requested",
  "classifier.classifications.0.classification.type": "binary",
  "classifier.classifications.0.classification.instructions": "\"Does the customer explicitly ask for a refund?\"",
  "classifier.classifications.0.classification.probability": 0.2,

  "classifier.classifications.1.classification.name": "department",
  "classifier.classifications.1.classification.type": "categorical",
  "classifier.classifications.1.classification.instructions": "\"Which team should own this ticket?\"",
  "classifier.classifications.1.classification.classes.0.class.label": "billing",
  "classifier.classifications.1.classification.classes.0.class.description": "\"Payments and refunds\"",
  "classifier.classifications.1.classification.classes.0.class.probability": 0.3,
  "classifier.classifications.1.classification.classes.1.class.label": "technical",
  "classifier.classifications.1.classification.classes.1.class.description": "\"Broken product workflows\"",
  "classifier.classifications.1.classification.classes.1.class.probability": 0.7,
  "classifier.classifications.1.classification.label": "technical",

  "classifier.classifications.2.classification.name": "urgency",
  "classifier.classifications.2.classification.type": "ordinal",
  "classifier.classifications.2.classification.instructions": "\"How urgently does this need attention?\"",
  "classifier.classifications.2.classification.classes.0.class.label": "routine",
  "classifier.classifications.2.classification.classes.0.class.value": 0.0,
  "classifier.classifications.2.classification.classes.0.class.probability": 0.0,
  "classifier.classifications.2.classification.classes.1.class.label": "soon",
  "classifier.classifications.2.classification.classes.1.class.value": 1.0,
  "classifier.classifications.2.classification.classes.1.class.probability": 0.2,
  "classifier.classifications.2.classification.classes.2.class.label": "today",
  "classifier.classifications.2.classification.classes.2.class.value": 2.0,
  "classifier.classifications.2.classification.classes.2.class.probability": 0.8,
  "classifier.classifications.2.classification.score": 1.8
}
```

| Task | Suggested rendering |
| --- | --- |
| Refund requested | **20% yes** probability bar; no inferred boolean decision |
| Department | **Technical** badge; Technical 70% / Billing 30% bars with criteria |
| Urgency | **1.8** marker on Routine (0) → Soon (1) → Today (2), with level probabilities |

## Mapping and handling

- TypeSafe Noul → `binary`; Choice → `categorical`; Score → `ordinal`. For Score,
  use legend keys as class labels, numeric keys as values, and legend entries as descriptions.
- Record one span per call. Preserve reported scores and probabilities; do not invent thresholds,
  confidence, or missing distributions. Confidence calculations may differ across producers.
- Capture definitions before execution so failed calls retain their inputs. `hide_inputs` masks
  content and task/class definitions; `hide_outputs` masks results and probabilities. Output
  labels that repeat hidden input definitions must also respect input masking.
